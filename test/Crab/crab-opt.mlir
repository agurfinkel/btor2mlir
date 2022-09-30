module  {
  func.func @main() -> i32 {
    %0 = crab.const 0 : i32
    %1 = crab.const 0 : i32
    cf.br ^bb1(%0, %1 : i32, i32)
  ^bb1(%2: i32, %3: i32):  // 2 preds: ^bb0, ^bb2
    %4 = crab.havoc() : i32
    crab.nd_br ^bb2(%2, %4 : i32, i32), ^bb3(%4, %3 : i32, i32)
  ^bb2(%5: i32, %6: i32):  // pred: ^bb1
    %7 = crab.const 9 : i32
    crab.assume ne(%5, %7) : i32
    %8 = crab.const 1 : i32
    %9 = crab.add(%2, %8) : i32
    %10 = crab.add(%3, %8) : i32
    cf.br ^bb1(%9, %10 : i32, i32)
  ^bb3(%11: i32, %12: i32):  // pred: ^bb1
    %13 = crab.const 10 : i32
    crab.assume uge(%11, %13) : i32
    crab.assert eq(%11, %12) : i32
    %14 = crab.const 0 : i32
    return %14 : i32
  }
}
