; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @__VERIFIER_error()

declare i64 @nd_bv64_st3()

declare void @btor2mlir_print_state_num(i64, i64, i64)

declare i64 @nd_bv64_st2()

define void @main() !dbg !3 {
  %1 = call i64 @nd_bv64_st2(), !dbg !7
  %2 = trunc i64 %1 to i33, !dbg !9
  %3 = call i64 @nd_bv64_st3(), !dbg !10
  %4 = trunc i64 %3 to i33, !dbg !11
  br label %5, !dbg !12

5:                                                ; preds = %19, %0
  %6 = phi i1 [ %16, %19 ], [ false, %0 ]
  %7 = phi i1 [ true, %19 ], [ false, %0 ]
  %8 = phi i33 [ 50, %19 ], [ %2, %0 ]
  %9 = phi i33 [ 50, %19 ], [ %4, %0 ]
  %10 = xor i33 %8, %9, !dbg !13
  %11 = xor i33 %10, -1, !dbg !14
  %12 = lshr i33 %11, 2, !dbg !15
  %13 = trunc i33 %12 to i1, !dbg !16
  %14 = xor i1 %13, true, !dbg !17
  %15 = and i1 %7, %14, !dbg !18
  %16 = and i1 %15, true, !dbg !19
  %17 = and i1 %6, true, !dbg !20
  %18 = xor i1 %17, true, !dbg !21
  br i1 %18, label %19, label %20, !dbg !22

19:                                               ; preds = %5
  br label %5, !dbg !23

20:                                               ; preds = %5
  call void @__VERIFIER_error(), !dbg !24
  unreachable, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 6, type: !5, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "btor2mlir-1/test/Btor/wolf-19C.btor.mlir.opt", directory: "/home/jetafese")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 8, column: 10, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 9, column: 10, scope: !8)
!10 = !DILocation(line: 10, column: 10, scope: !8)
!11 = !DILocation(line: 11, column: 10, scope: !8)
!12 = !DILocation(line: 12, column: 5, scope: !8)
!13 = !DILocation(line: 15, column: 11, scope: !8)
!14 = !DILocation(line: 17, column: 11, scope: !8)
!15 = !DILocation(line: 19, column: 11, scope: !8)
!16 = !DILocation(line: 20, column: 11, scope: !8)
!17 = !DILocation(line: 22, column: 11, scope: !8)
!18 = !DILocation(line: 23, column: 11, scope: !8)
!19 = !DILocation(line: 24, column: 11, scope: !8)
!20 = !DILocation(line: 26, column: 11, scope: !8)
!21 = !DILocation(line: 28, column: 11, scope: !8)
!22 = !DILocation(line: 29, column: 5, scope: !8)
!23 = !DILocation(line: 31, column: 5, scope: !8)
!24 = !DILocation(line: 33, column: 5, scope: !8)
!25 = !DILocation(line: 34, column: 5, scope: !8)
