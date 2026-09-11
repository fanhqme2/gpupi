#pragma once
#include <gmpxx.h>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <utility>
#include "host_arith.h"

// Host integers with 64-bit sizes. Only the public mpn layer sees large
// values; ordinary GMP mpz objects are used for small reference operations.
namespace wide {
static_assert(GMP_NUMB_BITS==64 && sizeof(mp_size_t)==8);
struct Rep {size_t _mp_alloc=0;int64_t _mp_size=0;mp_limb_t*_mp_d=nullptr;};
inline size_t mpz_size(const Rep*p){return p->_mp_size<0?-p->_mp_size:p->_mp_size;}
inline const mp_limb_t*mpz_limbs_read(const Rep*p){return p->_mp_d;}
inline void release(Rep*p){if(p->_mp_alloc){void(*f)(void*,size_t);mp_get_memory_functions(nullptr,nullptr,&f);f(p->_mp_d,p->_mp_alloc*8);}*p={};}
inline mp_limb_t*reserve(Rep*p,size_t n,bool preserve){
    if(n<=p->_mp_alloc)return p->_mp_d;
    if(n>SIZE_MAX/8-8)throw std::length_error("wide integer allocation overflow");
    // Carry limbs and short shifts must not copy a multi-gigabyte integer.
    size_t capacity=n+8;
    void*(*alloc)(size_t);mp_get_memory_functions(&alloc,nullptr,nullptr);
    auto*q=(mp_limb_t*)alloc(capacity*8);if(!q&&n)throw std::bad_alloc();
    auto size=p->_mp_size;if(preserve&&mpz_size(p))host_arith::copy(q,p->_mp_d,mpz_size(p)*8);
    release(p);*p={capacity,size,q};return q;
}
inline mp_limb_t*mpz_limbs_write(Rep*p,mp_size_t n){if(n<0)throw std::length_error("negative limb count");return reserve(p,n,false);}
inline mp_limb_t*mpz_limbs_modify(Rep*p,mp_size_t n){if(n<0)throw std::length_error("negative limb count");return reserve(p,n,true);}
inline void mpz_limbs_finish(Rep*p,mp_size_t n){int sign=n<0?-1:1;size_t count=n<0?-n:n;while(count&&!p->_mp_d[count-1])--count;p->_mp_size=sign*(int64_t)count;}
inline size_t mpz_sizeinbase(const Rep*p,int base){if(base!=2)throw std::invalid_argument("wide sizeinbase only supports base 2");size_t n=mpz_size(p);return n?(n-1)*64+64-__builtin_clzll(p->_mp_d[n-1]):1;}
inline int mpz_cmpabs(const Rep*a,const Rep*b){size_t an=mpz_size(a),bn=mpz_size(b);return an!=bn?(an>bn?1:-1):an?mpn_cmp(a->_mp_d,b->_mp_d,an):0;}
inline void assign(Rep*a,const Rep*b){if(a==b)return;size_t n=mpz_size(b);auto*d=reserve(a,n,false);if(n)host_arith::copy(d,b->_mp_d,n*8);a->_mp_size=b->_mp_size;}
inline __mpz_struct small(const Rep*p){if(mpz_size(p)>INT_MAX)throw std::length_error("large integer passed to GMP mpz reference operation");return {0,(int)p->_mp_size,p->_mp_d};}
inline void import(Rep*p,mpz_srcptr q){size_t n=::mpz_size(q);auto*d=reserve(p,n,false);if(n)host_arith::copy(d,::mpz_limbs_read(q),n*8);p->_mp_size=q->_mp_size;}
class Z {
    Rep value;
public:
    Z()=default;
    Z(long v){if(v){reserve(&value,1,false)[0]=v<0?0ull-(uint64_t)v:v;value._mp_size=v<0?-1:1;}}
    Z(unsigned long v){if(v){reserve(&value,1,false)[0]=v;value._mp_size=1;}}
    Z(int v):Z((long)v){} Z(unsigned v):Z((unsigned long)v){}
    template<class T,class U>Z(const __gmp_expr<T,U>&v){mpz_class x(v);import(&value,x.get_mpz_t());}
    Z(const Z&b){assign(&value,&b.value);}
    Z(Z&&b)noexcept:value(b.value){b.value={};}
    ~Z(){release(&value);}
    Z&operator=(Z b){std::swap(value,b.value);return *this;}
    Rep*get_mpz_t(){return &value;}const Rep*get_mpz_t()const{return &value;}
    static Z borrow(const Rep&p){Z z;z.value={0,p._mp_size,p._mp_d};return z;}
    void negate(){value._mp_size=-value._mp_size;}
    Z operator-()const{Z r=*this;r.value._mp_size=-r.value._mp_size;return r;}
    Z&operator+=(const Z&b);
    Z&operator-=(const Z&b);
    Z&operator<<=(uint64_t bits);
    Z&operator>>=(uint64_t bits);
    Z&operator++(){return *this+=Z(1);}Z&operator--(){return *this-=Z(1);}
    friend int compare(const Z&a,const Z&b){int as=mpz_sgn(&a.value),bs=mpz_sgn(&b.value);return as!=bs?(as>bs?1:-1):as*mpz_cmpabs(&a.value,&b.value);}
};
inline bool operator==(const Z&a,const Z&b){return compare(a,b)==0;}
inline bool operator!=(const Z&a,const Z&b){return compare(a,b)!=0;}
inline bool operator<(const Z&a,const Z&b){return compare(a,b)<0;}
inline bool operator>(const Z&a,const Z&b){return compare(a,b)>0;}
inline Z abs(const Z&a){return a<0?-a:a;}
inline void serial_add(Rep*a,const Rep*b,bool sub){
    if(a==b){Z copy=Z::borrow(*b);Z owned=copy;serial_add(a,owned.get_mpz_t(),sub);return;}
    size_t an=mpz_size(a),bn=mpz_size(b);int as=mpz_sgn(a),bs=mpz_sgn(b)*(sub?-1:1);
    if(!bn)return;if(!an){assign(a,b);a->_mp_size*=sub?-1:1;return;}
    bool diff=as!=bs,rev=diff&&mpz_cmpabs(a,b)<0;size_t n=std::max(an,bn);
    auto*out=reserve(a,n+1,true);if(n>an)std::memset(out+an,0,(n-an)*8);
    mp_limb_t carry=rev?mpn_sub(out,b->_mp_d,bn,out,an):diff?mpn_sub(out,out,n,b->_mp_d,bn):mpn_add(out,out,n,b->_mp_d,bn);
    out[n]=diff?0:carry;mpz_limbs_finish(a,(int64_t)(n+1)*(rev?bs:as));
}
inline Z&Z::operator+=(const Z&b){serial_add(&value,b.get_mpz_t(),false);return *this;}
inline Z&Z::operator-=(const Z&b){serial_add(&value,b.get_mpz_t(),true);return *this;}
inline Z operator+(Z a,const Z&b){host_arith::add(a,b);return a;}
inline Z operator-(Z a,const Z&b){host_arith::add(a,b,true);return a;}
inline void shift(Z&a,uint64_t bits,bool left){
    auto*p=a.get_mpz_t();size_t an=mpz_size(p);if(!bits||!an)return;
    int sign=mpz_sgn(p);size_t words=bits/64;unsigned part=bits%64;bool lost=false;
    if(!left&&sign<0){for(size_t i=0;i<std::min(an,words);i++)lost|=p->_mp_d[i]!=0;if(part&&words<an)lost|=(p->_mp_d[words]&((1ull<<part)-1))!=0;}
    if(!left&&words>=an){a=sign<0?Z(-1):Z();return;}
    size_t n=left?an+words+(part!=0):an-words;Z result;auto*out=mpz_limbs_write(result.get_mpz_t(),n);auto*src=p->_mp_d;
    host_arith::parallel(n,[&](size_t l,size_t r,size_t){for(size_t j=l;j<r;j++){
        if(left){if(j<words){out[j]=0;continue;}size_t i=j-words;uint64_t lo=i<an?src[i]:0,hi=i&&i-1<an?src[i-1]:0;out[j]=part?(lo<<part)|(hi>>(64-part)):lo;}
        else{size_t i=j+words;out[j]=part?(src[i]>>part)|((i+1<an?src[i+1]:0)<<(64-part)):src[i];}
    }});mpz_limbs_finish(result.get_mpz_t(),n);if(lost)++result;result.get_mpz_t()->_mp_size*=sign;a=std::move(result);
}
inline Z&Z::operator<<=(uint64_t n){shift(*this,n,true);return *this;}
inline Z&Z::operator>>=(uint64_t n){shift(*this,n,false);return *this;}
inline Z operator<<(Z a,uint64_t n){a<<=n;return a;}
inline Z operator>>(Z a,uint64_t n){a>>=n;return a;}
inline void mpz_add(Rep*c,const Rep*a,const Rep*b){Z copy,r=Z::borrow(*b);reserve(copy.get_mpz_t(),std::max(mpz_size(a),mpz_size(b))+1,false);assign(copy.get_mpz_t(),a);host_arith::add(copy,r);std::swap(*c,*copy.get_mpz_t());}
inline void mpz_mul(Rep*c,const Rep*a,const Rep*b){auto x=small(a),y=small(b);mpz_class z;::mpz_mul(z.get_mpz_t(),&x,&y);import(c,z.get_mpz_t());}
inline Z operator*(const Z&a,const Z&b){Z c;mpz_mul(c.get_mpz_t(),a.get_mpz_t(),b.get_mpz_t());return c;}
inline Z operator/(const Z&a,const Z&b){auto x=small(a.get_mpz_t()),y=small(b.get_mpz_t());mpz_class z;::mpz_tdiv_q(z.get_mpz_t(),&x,&y);return Z(z);}
inline Z operator&(const Z&a,const Z&b){auto x=small(a.get_mpz_t()),y=small(b.get_mpz_t());mpz_class z;::mpz_and(z.get_mpz_t(),&x,&y);return Z(z);}
inline void mpz_import(Rep*p,size_t n,int order,size_t size,int endian,size_t nails,const void*data){mpz_class z;::mpz_import(z.get_mpz_t(),n,order,size,endian,nails,data);import(p,z.get_mpz_t());}
inline void mpz_ui_pow_ui(Rep*p,unsigned long b,unsigned long e){mpz_class z;::mpz_ui_pow_ui(z.get_mpz_t(),b,e);import(p,z.get_mpz_t());}
inline void mpz_fdiv_qr(Rep*q,Rep*r,const Rep*a,const Rep*b){auto x=small(a),y=small(b);mpz_class u,v;::mpz_fdiv_qr(u.get_mpz_t(),v.get_mpz_t(),&x,&y);import(q,u.get_mpz_t());import(r,v.get_mpz_t());}
inline void mpz_setbit(Rep*p,mp_bitcnt_t bit){size_t n=bit/64+1,old=mpz_size(p);auto*d=reserve(p,std::max(n,old),true);if(n>old)std::memset(d+old,0,(n-old)*8);d[bit/64]|=1ull<<(bit%64);p->_mp_size=std::max(n,old);}
inline unsigned long mpz_fdiv_ui(const Rep*p,unsigned long m){
    if(!m)throw std::invalid_argument("zero modulus");size_t n=mpz_size(p);uint64_t r=0;
    struct Chunk{size_t n=0;uint64_t residue=0;};Chunk chunks[32];size_t count=0;
    if(n)host_arith::parallel(n,[&](size_t l,size_t h,size_t t){chunks[t]={h-l,mpn_mod_1(p->_mp_d+l,h-l,m)};});
    while(count<32&&chunks[count].n)++count;
    auto times=[&](uint64_t a,uint64_t b){return (uint64_t)((unsigned __int128)a*b%m);};
    uint64_t base=((unsigned __int128)1<<64)%m;
    while(count){auto c=chunks[--count];uint64_t power=1,b=base;for(size_t e=c.n;e;e>>=1,b=times(b,b))if(e&1)power=times(power,b);
        r=((unsigned __int128)r*power+c.residue)%m;
    }
    return p->_mp_size<0&&r?m-r:r;
}
}
