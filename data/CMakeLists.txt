
install(
  DIRECTORY atlas_eve_with_cere_type atlas_jakob_with_cere_type
  DESTINATION .
  USE_SOURCE_PERMISSIONS
  PATTERN ".svn" EXCLUDE
)
