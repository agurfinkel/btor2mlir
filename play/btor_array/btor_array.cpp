
#include <cassert>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/APInt.h>

#pragma clang diagnostic ignored "-Wstring-conversion"
#include <immer/map.hpp>

using namespace llvm;

struct APIntHasher {
  size_t operator()(const APInt &key) {
    return DenseMapInfo<APInt,void>::getHashValue(key);
  }
};

struct APIntEqualTo {
  bool operator()(const APInt &lhs, const APInt &rhs) {
    return DenseMapInfo<APInt,void>::isEqual(lhs, rhs);
  }
};


class BtorArray {
  unsigned m_idx_bw;
  unsigned m_elm_bw;
  APInt m_zero;

  using map_type = immer::map<APInt, APInt, APIntHasher, APIntEqualTo>;
  map_type m_data;
  llvm::Optional<APInt> m_init;

  public:
  BtorArray(const BtorArray&) = default;
  BtorArray& operator=(const BtorArray&) = default;
  BtorArray(BtorArray&&) = default;
  BtorArray& operator=(BtorArray&&) = default;

  BtorArray(unsigned idx_bw, unsigned elm_bw) : m_idx_bw(idx_bw), m_elm_bw(elm_bw), m_zero(m_elm_bw, 0) { }

  unsigned getIdxBitWidth() { return m_idx_bw; }
  unsigned getElmBitWidth() { return m_elm_bw; }

  BtorArray &setInit(APInt &&val) {
    assert(!m_init.hasValue());
    m_init.emplace(std::move(val));
    return *this;
  }

  BtorArray write(const APInt &key, const APInt &val) {
    assert(key.getBitWidth() == m_idx_bw);
    assert(val.getBitWidth() == m_elm_bw);
    BtorArray res(*this);
    res.m_data = res.m_data.set(key, val);
    return res;
  }

  const APInt& read(const APInt &key) {
    assert(key.getBitWidth() == m_idx_bw);
    if (m_data.count(key)) 
      return m_data.at(key);
    else if (m_init.hasValue()) 
       return m_init.getValue();
    else 
      return m_zero;
  }
};


int main(int argc, char **argv) {

    auto zero64 = APInt::getZero(64);
    auto ones64 = APInt::getAllOnes(64);

    BtorArray a1(32, 64);
    a1.setInit(APInt::getZero(64));

    auto key = APInt::getAllOnes(32);

    // should return default value
    assert(a1.read(key) == zero64);

    // new map with a different value
    auto a2 = a1.write(key, ones64);

    // new value changed
    assert(a2.read(key) == ones64);
    // old value unchanged
    assert(a1.read(key) == zero64);

    return 0;
}