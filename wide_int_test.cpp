#include "wide_int.h"
#include <cstdio>
#include <string>
using wide::Z;
static void require(bool value){if(!value)throw std::runtime_error("wide integer mismatch");}
static void equal(const Z&a,const mpz_class&b){Z expected(b);require(a==expected);}
int main(int argc,char**argv){
    gmp_randclass random(gmp_randinit_default);random.seed(817);
    for(int i=0;i<250;i++){
        mpz_class a=random.get_z_bits(100+i*733),b=random.get_z_bits(100+i*599);
        if(i&1)a=-a;if(i&2)b=-b;Z x(a),y(b);
        for(unsigned long p:{1ul,3ul,1000000007ul,18446744073709551557ul})require(mpz_fdiv_ui(x.get_mpz_t(),p)==mpz_fdiv_ui(a.get_mpz_t(),p));
        equal(x+y,mpz_class(a+b));equal(x-y,mpz_class(a-b));equal(x*y,mpz_class(a*b));
        Z z=x;host_arith::add(z,y);equal(z,mpz_class(a+b));z=x;host_arith::add(z,y,true);equal(z,mpz_class(a-b));
        for(unsigned shift:{0u,1u,63u,64u,65u,unsigned(i*757)}){equal(x<<shift,mpz_class(a<<shift));equal(x>>shift,mpz_class(a>>shift));}
        auto av=*x.get_mpz_t(),bv=*y.get_mpz_t();Z sum;mpz_add(sum.get_mpz_t(),&av,&bv);equal(sum,mpz_class(a+b));
        z=x;host_arith::add(z,z);equal(z,mpz_class(a*2));host_arith::add(z,z,true);require(z==0);
        if(b!=0)equal(x/y,mpz_class(a/b));
    }
    if(argc>1&&std::string(argv[1])=="--large"){
        uint64_t bit=((1ull<<31)+17)*64+7;Z x;mpz_setbit(x.get_mpz_t(),bit);++x;
        require(mpz_size(x.get_mpz_t())==(1ull<<31)+18);require(mpz_sizeinbase(x.get_mpz_t(),2)==bit+1);
        Z y=x;host_arith::shift(y,65);host_arith::shift(y,-65);require(y==x);
        host_arith::add(y,x);Z twice=x;host_arith::shift(twice,1);require(y==twice);twice=Z();
        host_arith::add(y,x,true);require(y==x);
        constexpr unsigned long p=1000000007;mpz_class expected;mpz_ui_pow_ui(expected.get_mpz_t(),2,1);mpz_class e(bit),m(p);mpz_powm(expected.get_mpz_t(),expected.get_mpz_t(),e.get_mpz_t(),m.get_mpz_t());
        require(mpz_fdiv_ui(x.get_mpz_t(),p)==(mpz_get_ui(expected.get_mpz_t())+1)%p);
        printf("PASS: 64-bit host integers above INT_MAX limbs (%llu bits)\n",(unsigned long long)bit);
    }
    puts("PASS: wide integer arithmetic against GMP");
}
