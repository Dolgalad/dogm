cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(boost_cplex_example)

set(CMAKE_MODULE_PATH               "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS                 "${CMAKE_CXX_FLAGS} -std=c++17 -DIL_STD -m64 -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG           "${CMAKE_CXX_FLAGS_DEBUG} -Og -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -flto")
set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX   "-isystem ")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(LINKER_OPTIONS                  -flto -Wl,--no-as-needed)

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(CMAKE_CXX_FLAGS_DEBUG       "${CMAKE_CXX_FLAGS_DEBUG} -save-temps=obj")
endif()


find_package(CPLEX)
include_directories(SYSTEM ${CPLEX_INCLUDE_DIRS})

find_package(Boost 1.85)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

ADD_EXECUTABLE(torch_boost_cplex_example torch_boost_cplex_example.cpp )

target_link_libraries(torch_boost_cplex_example PRIVATE ${CPLEX_LIBRARIES})
target_link_libraries(torch_boost_cplex_example PRIVATE dl)
TARGET_LINK_LIBRARIES(torch_boost_cplex_example LINK_PUBLIC ${Boost_LIBRARIES} )
target_link_libraries(torch_boost_cplex_example PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET torch_boost_cplex_example PROPERTY CXX_STANDARD 17)

ADD_EXECUTABLE(read_metis_graph read_metis_graph.cpp )
TARGET_LINK_LIBRARIES(read_metis_graph LINK_PUBLIC ${Boost_LIBRARIES} )

ADD_EXECUTABLE(test_dataset test_dataset.cpp )
target_link_libraries(test_dataset PRIVATE "${TORCH_LIBRARIES}")
TARGET_LINK_LIBRARIES(torch_boost_cplex_example LINK_PUBLIC ${Boost_LIBRARIES})

#add_library(LibsMIS SHARED mis.cpp)
#target_link_libraries(LibsMIS PRIVATE ${CPLEX_LIBRARIES})
#target_link_libraries(LibsMIS PRIVATE dl)
#
#ADD_EXECUTABLE(mis_er_dataset mis_er_dataset.cpp)
#target_link_libraries(mis_er_dataset PRIVATE ${Boost_LIBRARIES})
#target_link_libraries(mis_er_dataset PRIVATE ${CPLEX_LIBRARIES})
#target_link_libraries(mis_er_dataset PRIVATE dl)

#add_executable(         cplex_test cplex_test.cpp test_lib.cpp )
add_executable(         mis_er_dataset mis_er_dataset.cpp mis.cpp )
target_link_libraries(  mis_er_dataset LINK_PUBLIC ${BOOST_LIBRARIES})
target_link_libraries(  mis_er_dataset PRIVATE ${CPLEX_LIBRARIES})
target_link_libraries(  mis_er_dataset PRIVATE dl)

add_executable(mis_train mis_train.cpp belief_propagation.cpp)
target_link_libraries(mis_train PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET mis_train PROPERTY CXX_STANDARD 17)

add_executable(test_mplbp test_mplbp.cpp belief_propagation.cpp)
target_link_libraries(test_mplbp PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET test_mplbp PROPERTY CXX_STANDARD 17)

