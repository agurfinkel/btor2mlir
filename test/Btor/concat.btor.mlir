module {
  func @main() {
    %0 = btor.constant 0 : i8
    br ^bb1(%0 : i8)
  ^bb1(%1: i8):  // 2 preds: ^bb0, ^bb1
    %2 = btor.constant false
    %3 = btor.constant -1 : i7
    %4 = btor.concat %3, %2 : i7, i1, i8
    %5 = btor.constant -2 : i8
    %6 = btor.cmp eq, %1, %5 : i8
    btor.assert_not(%6)
    br ^bb1(%4 : i8)
  }
}
