module {
  func @main() {
    %0 = btor.constant false
    %1 = btor.nd_state 2 : i33
    %2 = btor.nd_state 3 : i33
    br ^bb1(%0, %0, %1, %2 : i1, i1, i33, i33)
  ^bb1(%3: i1, %4: i1, %5: i33, %6: i33):  // 2 preds: ^bb0, ^bb1
    %7 = btor.constant true
    %8 = btor.xnor %5, %6 : i33
    %9 = btor.constant 2 : i33
    %10 = btor.constant 2 : i33
    %11 = btor.slice %8, %9, %10 : i33, i1
    %12 = btor.not %11 : i1
    %13 = btor.and %4, %12 : i1
    %14 = btor.and %13, %7 : i1
    %15 = btor.constant 50 : i33
    %16 = btor.and %3, %7 : i1
    btor.assert_not(%16)
    br ^bb1(%14, %7, %15, %15 : i1, i1, i33, i33)
  }
}
