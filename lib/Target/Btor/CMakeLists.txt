add_mlir_dialect_library(MLIRBtorTranslate
        BtorToBtorIRTranslation.cpp
        btor2tools/btorsimhelpers.cpp
        btor2tools/btorsimstate.cpp
        btor2tools/btorsimvcd.cpp
        btor2tools/btorsimam.cpp
        btor2tools/btorsimrng.c
        btor2tools/btorsimbv.c
        btor2tools/btor2parser/btor2parser.c

	LINK_LIBS PUBLIC
	MLIRIR
	)