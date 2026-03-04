file(GLOB_RECURSE ES_SHOWCASE CONFIGURE_DEPENDS "*.cpp")
add_library(es_showcase OBJECT ${ES_SHOWCASE})
target_link_libraries(es_showcase PUBLIC es_all graph graph_base c_sec)
target_compile_options(es_showcase PRIVATE -g -O0)