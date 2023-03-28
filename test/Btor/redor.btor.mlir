module {
  func @main() {
    %0 = btor.constant false
    br ^bb1(%0 : i1)
  ^bb1(%1: i1):  // 2 preds: ^bb0, ^bb1
    %2 = btor.input 3 : i8
    %3 = btor.redor %2 : i8
    btor.assert_not(%1)
    br ^bb1(%3 : i1)
  }
}
