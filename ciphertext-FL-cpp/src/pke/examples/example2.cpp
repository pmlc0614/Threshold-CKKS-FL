#include "openfhe.h"

using namespace lbcrypto;
using namespace std;

int main() {

    BigInteger a(32);
    BigInteger b(54);
    BigInteger m(7);

    std::cout << a.ModAdd(b,m) << std::endl;
    std::cout << a.ModSub(b,m) << std::endl;
    std::cout << a.ModMul(b,m) << std::endl;

    // BigVector c(10, 5);
    // std::cout << c << std::endl;

    // BigVector d(10, 7, {1,2,3,4,5});
    // BigVector e(10, 7, {19,20,13,9,3});
    // cout << d+e << endl;

    NativeInteger p(32);
    NativeInteger q(54);
    NativeInteger l(7);
    
    cout << p.ModAdd(q, l) << endl;
    cout << p.ModSub(q, l) << endl;
    cout << p.ModMul(q, l) << endl;

    NativeVector d(10, 7, {1,2,3,4,5});
    NativeVector e(10, 7, {19,20,13,9,3});
    cout << d+e << endl;

    return 0;

}