cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(dogm)

set(CMAKE_MODULE_PATH               "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS                 "${CMAKE_CXX_FLAGS} -std=c++17 -DIL_STD -m64 -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG           "${CMAKE_CXX_FLAGS_DEBUG} -Og -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -flto")
set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX   "-isystem ")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(LINKER_OPTIONS                  -flto -Wl,--no-as-needed)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(CMAKE_CXX_FLAGS_DEBUG       "${CMAKE_CXX_FLAGS_DEBUG} -save-temps=obj")
endif()


find_package(CPLEX)
if(CPLEX_FOUND)
    include_directories(SYSTEM ${CPLEX_INCLUDE_DIRS})
endif()

find_package(Boost 1.85)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(CPLEX_FOUND AND Boost_FOUND)
    add_executable(torch_boost_cplex_example torch_boost_cplex_example.cpp )
    target_link_libraries(torch_boost_cplex_example PRIVATE ${CPLEX_LIBRARIES})
    target_link_libraries(torch_boost_cplex_example PRIVATE dl)
    target_link_libraries(torch_boost_cplex_example LINK_PUBLIC ${Boost_LIBRARIES} )
    target_link_libraries(torch_boost_cplex_example PRIVATE "${TORCH_LIBRARIES}")
    set_property(TARGET torch_boost_cplex_example PROPERTY CXX_STANDARD 17)

    add_executable(         mis_er_dataset mis_er_dataset.cpp mis.cpp )
    target_link_libraries(  mis_er_dataset LINK_PUBLIC ${Boost_LIBRARIES})
    target_link_libraries(  mis_er_dataset PRIVATE ${CPLEX_LIBRARIES})
    target_link_libraries(  mis_er_dataset PRIVATE dl)

endif()

if(Boost_FOUND)
    add_executable(read_metis_graph read_metis_graph.cpp )
    target_link_libraries(read_metis_graph LINK_PUBLIC ${Boost_LIBRARIES} )

    add_executable(test_dataset test_dataset.cpp )
    target_link_libraries(test_dataset PRIVATE "${TORCH_LIBRARIES}")
    target_link_libraries(test_dataset LINK_PUBLIC ${Boost_LIBRARIES})

endif()


add_library(BeliefPropagation SHARED belief_propagation.cpp)
target_link_libraries(BeliefPropagation PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET BeliefPropagation PROPERTY CXX_STANDARD 17)

#add_executable(mis_train mis_train.cpp)
#target_link_libraries(mis_train PRIVATE "${TORCH_LIBRARIES}")
#set_property(TARGET mis_train PROPERTY CXX_STANDARD 17)
#target_link_libraries(mis_train PUBLIC BeliefPropagation)
#
#
#add_executable(mis_train_lazy mis_train_lazy.cpp)
#target_link_libraries(mis_train_lazy PRIVATE "${TORCH_LIBRARIES}")
#set_property(TARGET mis_train_lazy PROPERTY CXX_STANDARD 17)
#target_link_libraries(mis_train_lazy PUBLIC BeliefPropagation)
#
#add_executable(mis_train_nocopy mis_train_nocopy.cpp)
#target_link_libraries(mis_train_nocopy PRIVATE "${TORCH_LIBRARIES}")
#set_property(TARGET mis_train_nocopy PROPERTY CXX_STANDARD 17)
#target_link_libraries(mis_train_nocopy PUBLIC BeliefPropagation)
#
#add_executable(mis_train_lazy_nocopy mis_train_lazy_nocopy.cpp)
#target_link_libraries(mis_train_lazy_nocopy PRIVATE "${TORCH_LIBRARIES}")
#set_property(TARGET mis_train_lazy_nocopy PROPERTY CXX_STANDARD 17)
#target_link_libraries(mis_train_lazy_nocopy PUBLIC BeliefPropagation)
#
#add_executable(chkpt2cpu chkpt2cpu.cpp)
#target_link_libraries(chkpt2cpu PRIVATE "${TORCH_LIBRARIES}")
#set_property(TARGET chkpt2cpu PROPERTY CXX_STANDARD 17)
#
#add_executable(test_mplbp test_mplbp.cpp)
#target_link_libraries(test_mplbp PRIVATE "${TORCH_LIBRARIES}")
#set_property(TARGET test_mplbp PROPERTY CXX_STANDARD 17)
#target_link_libraries(test_mplbp PUBLIC BeliefPropagation)

