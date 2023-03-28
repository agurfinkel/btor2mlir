; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @verifier.error()

define void @main() !dbg !3 {
  br label %1, !dbg !7

1:                                                ; preds = %5, %0
  %2 = phi i8 [ -2, %5 ], [ 0, %0 ]
  %3 = icmp eq i8 %2, -2, !dbg !9
  %4 = xor i1 %3, true, !dbg !10
  br i1 %4, label %5, label %6, !dbg !11

5:                                                ; preds = %1
  br label %1, !dbg !12

6:                                                ; preds = %1
  call void @verifier.error(), !dbg !13
  unreachable, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 3, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "../../../../btor2mlir-1/test/Btor/concat.btor.mlir.opt", directory: "/home/jetafese/runbench/svcomp/sea/btor2mlir")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 5, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 15, column: 11, scope: !8)
!10 = !DILocation(line: 17, column: 11, scope: !8)
!11 = !DILocation(line: 18, column: 5, scope: !8)
!12 = !DILocation(line: 20, column: 5, scope: !8)
!13 = !DILocation(line: 22, column: 5, scope: !8)
!14 = !DILocation(line: 23, column: 5, scope: !8)
