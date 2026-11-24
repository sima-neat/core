#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <knownfolders.h>
#include <shlobj.h>
#include <strsafe.h>
#include <wchar.h>

static const wchar_t kRegistryKey[] = L"Software\\SiMa.ai\\NeatPcieRollbackFixture";
static const wchar_t kRegistryValue[] = L"Sentinel";
static const wchar_t kExpectedRegistryValue[] = L"owned-by-windows-installer";
static const wchar_t kSentinelName[] = L"rollback-sentinel.txt";
static const wchar_t kObservationName[] =
    L"SiMaNeatMsiRollbackObserved-5C5FB61E-7355-4B72-87FD-5FDD4CC2715B.txt";

static int require_sibling_sentinel(void) {
  wchar_t path[MAX_PATH];
  DWORD length = GetModuleFileNameW(NULL, path, ARRAYSIZE(path));
  if (length == 0 || length >= ARRAYSIZE(path)) {
    return 41;
  }
  wchar_t* separator = wcsrchr(path, L'\\');
  if (separator == NULL) {
    return 42;
  }
  separator[1] = L'\0';
  if (FAILED(StringCchCatW(path, ARRAYSIZE(path), kSentinelName))) {
    return 43;
  }
  return GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES ? 44 : 0;
}

static int require_registry_sentinel(void) {
  HKEY key = NULL;
  LONG status =
      RegOpenKeyExW(HKEY_LOCAL_MACHINE, kRegistryKey, 0, KEY_QUERY_VALUE | KEY_WOW64_64KEY, &key);
  if (status != ERROR_SUCCESS) {
    return 51;
  }
  wchar_t value[64];
  DWORD type = 0;
  DWORD size = sizeof(value);
  status = RegQueryValueExW(key, kRegistryValue, NULL, &type, (LPBYTE)value, &size);
  RegCloseKey(key);
  if (status != ERROR_SUCCESS || type != REG_SZ || wcscmp(value, kExpectedRegistryValue) != 0) {
    return 52;
  }
  return 0;
}

static int write_observation(void) {
  PWSTR common_application_data = NULL;
  HRESULT result =
      SHGetKnownFolderPath(&FOLDERID_ProgramData, KF_FLAG_DEFAULT, NULL, &common_application_data);
  if (FAILED(result) || common_application_data == NULL) {
    CoTaskMemFree(common_application_data);
    return 61;
  }
  wchar_t path[MAX_PATH];
  result =
      StringCchPrintfW(path, ARRAYSIZE(path), L"%s\\%s", common_application_data, kObservationName);
  CoTaskMemFree(common_application_data);
  if (FAILED(result)) {
    return 62;
  }
  HANDLE output =
      CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
  if (output == INVALID_HANDLE_VALUE) {
    return 63;
  }
  static const char observation[] = "schema_version=1\r\nfile=present\r\nregistry=present\r\n";
  DWORD written = 0;
  BOOL ok = WriteFile(output, observation, (DWORD)(sizeof(observation) - 1), &written, NULL);
  CloseHandle(output);
  return ok && written == sizeof(observation) - 1 ? 0 : 64;
}

int wmain(void) {
  int status = require_sibling_sentinel();
  if (status != 0) {
    return status;
  }
  status = require_registry_sentinel();
  if (status != 0) {
    return status;
  }
  status = write_observation();
  if (status != 0) {
    return status;
  }
  // Any non-zero exit from this deferred, checked custom action makes Windows
  // Installer execute its rollback script. The executable is CI-only.
  return ERROR_INSTALL_FAILURE;
}
