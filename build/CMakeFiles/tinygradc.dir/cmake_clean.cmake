file(REMOVE_RECURSE
  "libtinygradc.a"
  "libtinygradc.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/tinygradc.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
