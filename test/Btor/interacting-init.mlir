module {
  func @main() {
    %0 = btor.constant 0 : i8
    %1 = btor.input 0 : i8
    %2 = btor.add %0, %1 : i8
    %3 = btor.nd_state 2 : i8
    %4 = btor.and %3, %0 : i8
    br ^bb1(%0, %2, %3, %4 : i8, i8, i8, i8)
  ^bb1(%5: i8, %6: i8, %7: i8, %8: i8):  // 2 preds: ^bb0, ^bb1
    %9 = btor.input 0 : i8
    %10 = btor.add %5, %9 : i8
    %11 = btor.cmp eq, %5, %10 : i8
    btor.assert_not(%11)
    %12 = btor.nd_state 2 : i8
    br ^bb1(%5, %6, %12, %8 : i8, i8, i8, i8)
  }
}
