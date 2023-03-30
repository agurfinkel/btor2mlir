; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @__VERIFIER_error()

define void @main() !dbg !3 {
  br label %1, !dbg !7

1:                                                ; preds = %2, %0
  br i1 false, label %2, label %3, !dbg !9

2:                                                ; preds = %1
  br label %1, !dbg !10

3:                                                ; preds = %1
  call void @__VERIFIER_error(), !dbg !11
  unreachable, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 3, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "btor2mlir-1/test/Btor/mod_zero.btor.mlir.opt", directory: "/home/jetafese")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 24, column: 5, scope: !8)
!10 = !DILocation(line: 26, column: 5, scope: !8)
!11 = !DILocation(line: 28, column: 5, scope: !8)
!12 = !DILocation(line: 29, column: 5, scope: !8)
