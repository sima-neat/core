# Windows Installer owns native libraries, headers, CMake package files and
# Neat runtime assets. Python remains a separately owned wheel.
set(CPACK_GENERATOR "WIX")
set(CPACK_PACKAGE_NAME "sima-pcie-host")
set(SIMAPCIE_WIX_INSTALL_DIRECTORY "SimaPCIeHost" CACHE STRING
  "Windows Installer destination folder below Program Files")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${SIMAPCIE_WIX_INSTALL_DIRECTORY}")
set(CPACK_PACKAGE_VENDOR "SiMa.ai")
set(CPACK_PACKAGE_CONTACT "support@sima.ai")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Host-side NEAT PCIe coprocessor runtime")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
configure_file("${SIMA_NEAT_CORE_ROOT}/LICENSE"
  "${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt" COPYONLY)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt")
set(CPACK_WIX_VERSION 3)
set(CPACK_WIX_ARCHITECTURE x64)
set(CPACK_WIX_INSTALL_SCOPE perMachine)
set(CPACK_WIX_UPGRADE_GUID "1A974787-A106-46EB-A5E7-36FD0F66519E")
set(CPACK_WIX_CMAKE_PACKAGE_REGISTRY "SimaPCIeHost")
set(CPACK_WIX_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
set(CPACK_COMPONENTS_ALL Runtime Development PcieHostTutorials)
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "SiMa PCIe host runtime")
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "SiMa PCIe host development files")
set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS Runtime)
set(CPACK_COMPONENT_PCIEHOSTTUTORIALS_DISPLAY_NAME "SiMa PCIe host tutorials")
set(CPACK_COMPONENT_PCIEHOSTTUTORIALS_DEPENDS Runtime)

# Like the DEB's exact Scheduler Depends, the MSI refuses a first installation
# unless the exact Scheduler MSI is installed. build_windows.py reads both
# values from the selected Scheduler MSI's Property table.
set(SIMAPCIE_SCHEDULER_PACKAGE_VERSION "" CACHE STRING
  "Exact selected native Scheduler MSI ProductVersion")
set(SIMAPCIE_SCHEDULER_WIX_UPGRADE_CODE "" CACHE STRING
  "UpgradeCode of the selected native Scheduler MSI")
set(_native_package_guard
  "message(FATAL_ERROR \"Native Core packaging requires its selected Scheduler MSI version and UpgradeCode\")")
if(SIMAPCIE_SCHEDULER_PACKAGE_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$"
    AND SIMAPCIE_SCHEDULER_WIX_UPGRADE_CODE MATCHES "^{[0-9A-F-]+}$")
  set(CPACK_WIX_CANDLE_EXTRA_FLAGS
    "-dSimaSchedulerVersion=${SIMAPCIE_SCHEDULER_PACKAGE_VERSION}"
    "-dSimaSchedulerUpgradeCode=${SIMAPCIE_SCHEDULER_WIX_UPGRADE_CODE}")
  set(_native_package_guard "")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/NativePackageGuard.cmake" "${_native_package_guard}\n")
set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/NativePackageGuard.cmake")
set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/SchedulerDependency.wxs.xml")
include(CPack)
