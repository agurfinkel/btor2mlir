configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(BTOR_TEST_DEPENDS
        FileCheck count not
        btor2mlir-opt
        btor2mlir-translate
        )

add_lit_testsuite(check-btor2mlir "Running the btor regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${BTOR_TEST_DEPENDS}
        )
set_target_properties(check-btor2mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(BTOR2MLIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BTOR_TEST_DEPENDS})
