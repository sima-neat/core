function(simapcie_add_example name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE sima_neat_pcie_host)
endfunction()

simapcie_add_example(simapcie_tensor_fast_paths tensor_fast_paths.cpp)
simapcie_add_example(simapcie_tensor_slow_path tensor_slow_path.cpp)
