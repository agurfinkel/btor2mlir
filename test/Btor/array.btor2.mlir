module {
  func @main() {
    %0 = btor.constant 1 : i4
    %1 = btor.array %0 : vector<16xi4>
    br ^bb1(%1 : vector<16xi4>)
  ^bb1(%2: vector<16xi4>):  // 2 preds: ^bb0, ^bb1
    %3 = btor.constant 1 : i4
    %4 = btor.constant -8 : i4
    %5 = btor.read %2[%4] : vector<16xi4>, i4
    %6 = btor.add %5, %3 : i4
    %7 = btor.write %6, %2[%4] : vector<16xi4>
    %8 = btor.constant -1 : i4
    %9 = btor.cmp eq, %5, %8 : i4
    btor.assert_not(%9)
    br ^bb1(%7 : vector<16xi4>)
  }
}
