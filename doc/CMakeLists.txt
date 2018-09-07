project(MERCATOR_MANUAL)

cmake_minimum_required(VERSION 3.8)

add_custom_command(OUTPUT mercator-manual.pdf 
                  COMMAND pdflatex -halt-on-error ${CMAKE_CURRENT_SOURCE_DIR}/mercator-manual.tex >/dev/null
		  DEPENDS mercator-manual.tex
		  COMMENT "Building manual")

add_custom_target(manual ALL
		  DEPENDS mercator-manual.pdf)

set(EXTRA_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/mercator-manual.aux 
    ${CMAKE_CURRENT_BINARY_DIR}/mercator-manual.log)

set_directory_properties(PROPERTIES
  ADDITIONAL_MAKE_CLEAN_FILES "${EXTRA_FILES}"
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mercator-manual.pdf DESTINATION doc)
