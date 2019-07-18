


add_subdirectory(cpp)

if (OMPI_FOUND)
    #add_subdirectory(ompi)
endif()

if (BOOST_FOUND)
   # add_subdirectory(boost)
endif()

if(BOOST_FOUND AND OMPI_FOUND)
    # add_subdirectory(boost_ompi)
endif()
