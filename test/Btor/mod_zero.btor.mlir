module {
  func @main() {
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %0 = btor.constant 6 : i5
    %1 = btor.constant 0 : i5
    %2 = btor.smod %0, %1 : i5
    %3 = btor.cmp eq, %2, %0 : i5
    btor.assert_not(%3)
    br ^bb1
  }
}
