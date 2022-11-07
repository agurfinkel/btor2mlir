module {
  func.func @main() {
    %0 = btor.constant 1 : i4
    cf.br ^bb1(%0, %0 : i4, i4)
  ^bb1(%1: i4, %2: i4):  // 2 preds: ^bb0, ^bb1
    %3 = btor.constant 1 : i4
    %4 = btor.add %2, %3 : i4
    %5 = btor.mul %1, %2 : i4
    %6 = btor.constant -1 : i4
    %7 = btor.cmp eq, %2, %6 : i4
    btor.assert_not(%7)
    %8 = btor.constant 0 : i4
    %9 = btor.constant 0 : i4
    %10 = btor.slice %1, %8, %9 : i4, i1
    %11 = btor.constant 3 : i4
    %12 = btor.cmp ugt, %2, %11 : i4
    %13 = btor.and %12, %10 : i1
    btor.assert_not(%13)
    cf.br ^bb1(%5, %4 : i4, i4)
  }
}
