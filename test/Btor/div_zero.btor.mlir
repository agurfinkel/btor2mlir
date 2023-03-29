module {
  func @main() {
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %0 = btor.constant -1 : i5
    %1 = btor.constant 0 : i5
    %2 = btor.constant 6 : i5
    %3 = btor.udiv %2, %1 : i5
    %4 = btor.cmp eq, %3, %0 : i5
    btor.assert_not(%4)
    br ^bb1
  }
}
