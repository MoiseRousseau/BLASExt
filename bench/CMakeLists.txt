add_executable(dvdvt bench_dvdvt.c)
target_link_libraries(dvdvt libBLASExt m)

add_executable(zvdvh bench_zvdvh.c)
target_link_libraries(zvdvh libBLASExt m)
