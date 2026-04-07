/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mmpa_stub.h"
#include <dirent.h>
#include "mmpa/mmpa_api.h"
#include "framework/common/taskdown_common.h"
#include <string>
#include <atomic>

namespace {
  std::atomic<bool> g_program_exiting{false};
  struct ExitGuard {
    ~ExitGuard() {
        g_program_exiting.store(true, std::memory_order_relaxed);
    }
  };
  static ExitGuard g_exit_guard; // 静态对象，其析构函数会设置退出标志
}


#ifdef __cplusplus
extern "C" {
#endif

typedef int mmErrorMSg;
#define MMPA_MAX_SLEEP_MILLSECOND_USING_USLEEP 1000
#define MMPA_MSEC_TO_USEC 1000
#define MMPA_MAX_SLEEP_MICROSECOND_USING_USLEEP 1000000

INT32 mmOpen(const CHAR *path_name, INT32 flags) {
  INT32 fd = HANDLE_INVALID_VALUE;

  if (NULL == path_name) {
    syslog(LOG_ERR, "The path name pointer is null.\r\n");
    return EN_INVALID_PARAM;
  }
  if ((flags != O_RDONLY) && (0 == (flags & (O_WRONLY | O_RDWR | O_CREAT)))) {
    syslog(LOG_ERR, "The file open mode is error.\r\n");
    return EN_INVALID_PARAM;
  }

  fd = open(path_name, flags, S_IRWXU | S_IRWXG);
  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "Open file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return fd;
}

INT32 mmOpen2(const CHAR *path_name, INT32 flags, MODE mode) {
  INT32 fd = HANDLE_INVALID_VALUE;

  if (NULL == path_name) {
    syslog(LOG_ERR, "The path name pointer is null.\r\n");
    return EN_INVALID_PARAM;
  }
  if (MMPA_ZERO == (flags & (O_RDONLY | O_WRONLY | O_RDWR | O_CREAT))) {
    syslog(LOG_ERR, "The file open mode is error.\r\n");
    return EN_INVALID_PARAM;
  }
  if ((MMPA_ZERO == (mode & (S_IRUSR | S_IREAD))) && (MMPA_ZERO == (mode & (S_IWUSR | S_IWRITE)))) {
    syslog(LOG_ERR, "The permission mode of the file is error.\r\n");
    return EN_INVALID_PARAM;
  }

  fd = open(path_name, flags, mode);
  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "Open file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return fd;
}

INT32 mmClose(INT32 fd) {
  INT32 result = EN_OK;

  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "The file fd is invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = close(fd);
  if (EN_OK != result) {
    syslog(LOG_ERR, "Close the file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return EN_OK;
}

mmSsize_t mmWrite(INT32 fd, VOID *mm_buf, UINT32 mm_count) {
  mmSsize_t result = MMPA_ZERO;

  if ((fd < MMPA_ZERO) || (NULL == mm_buf)) {
    syslog(LOG_ERR, "Input parameter invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = write(fd, mm_buf, (size_t)mm_count);
  if (result < MMPA_ZERO) {
    syslog(LOG_ERR, "Write buf to file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return result;
}

mmSsize_t mmRead(INT32 fd, VOID *mm_buf, UINT32 mm_count) {
  mmSsize_t result = MMPA_ZERO;

  if ((fd < MMPA_ZERO) || (NULL == mm_buf)) {
    syslog(LOG_ERR, "Input parameter invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = read(fd, mm_buf, (size_t)mm_count);
  if (result < MMPA_ZERO) {
    syslog(LOG_ERR, "Read file to buf failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return result;
}

VOID *mmMmap(mmFd_t fd, mmSize_t size, mmOfft_t offset, mmFd_t *extra, INT32 prot, INT32 flags) {
  return mmap(nullptr, size, prot, flags, fd, offset);
}

INT32 mmMunMap(VOID *data, mmSize_t size, mmFd_t *extra) {
  return munmap(data, size);
}

INT32 mmFStatGet(INT32 fd, mmStat_t *buf) {
  return fstat(fd, buf);
}

INT32 mmMkdir(const CHAR *lp_path_name, mmMode_t mode) {
  INT32 t_mode = mode;
  INT32 ret = EN_OK;

  if (NULL == lp_path_name) {
    syslog(LOG_ERR, "The input path is null.\r\n");
    return EN_INVALID_PARAM;
  }

  if (t_mode < MMPA_ZERO) {
    syslog(LOG_ERR, "The input mode is wrong.\r\n");
    return EN_INVALID_PARAM;
  }

  ret = mkdir(lp_path_name, mode);

  if (EN_OK != ret) {
    syslog(LOG_ERR, "Failed to create the directory, the ret is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmRmdir(const CHAR *lp_path_name) {
  return rmdir(lp_path_name);
}

mmTimespec mmGetTickCount() {
  mmTimespec rts;
  struct timespec ts = {0};
  (void)clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  rts.tv_sec = ts.tv_sec;
  rts.tv_nsec = ts.tv_nsec;
  return rts;
}

INT32 mmGetTid() {
  INT32 ret = (INT32)syscall(SYS_gettid);

  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }

  return ret;
}

INT32 mmGetSystemTime(mmSystemTime_t *sysTime) {
  // Beijing olympics
  sysTime->wYear = 2008;
  sysTime->wMonth = 8;
  sysTime->wDay = 8;
  sysTime->wHour = 20;
  sysTime->wMinute = 8;
  sysTime->wSecond = 0;
  return 0;
}

INT32 mmAccess(const CHAR *path_name) {
  if (path_name == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = access(path_name, F_OK);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmStatGet(const CHAR *path, mmStat_t *buffer) {
  if ((path == NULL) || (buffer == NULL)) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = stat(path, buffer);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmGetFileSize(const CHAR *file_name, ULONGLONG *length) {
  if ((file_name == NULL) || (length == NULL)) {
    return EN_INVALID_PARAM;
  }
  struct stat file_stat;
  (void)memset_s(&file_stat, sizeof(file_stat), 0, sizeof(file_stat));  // unsafe_function_ignore: memset
  INT32 ret = lstat(file_name, &file_stat);
  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }
  *length = (ULONGLONG)file_stat.st_size;
  return EN_OK;
}

INT32 mmScandir(const CHAR *path, mmDirent ***entryList, mmFilter filterFunc,  mmSort sort)
{
  if ((path == NULL) || (entryList == NULL)) {
    return EN_INVALID_PARAM;
  }
  INT32 count = scandir(path, entryList, filterFunc, sort);
  if (count < MMPA_ZERO) {
    return EN_ERROR;
  }
  return count;
}

VOID mmScandirFree(mmDirent **entryList, INT32 count)
{
  if (entryList == NULL) {
    return;
  }
  INT32 j;
  for (j = 0; j < count; j++) {
    if (entryList[j] != NULL) {
      free(entryList[j]);
      entryList[j] = NULL;
    }
  }
  free(entryList);
}

INT32 mmAccess2(const CHAR *pathName, INT32 mode)
{
  if (pathName == NULL) {
    return EN_INVALID_PARAM;
  }
  INT32 ret = access(pathName, mode);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmGetTimeOfDay(mmTimeval *timeVal, mmTimezone *timeZone)
{
  return 0;
}

INT32 mmRealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen)
{
  if (path == nullptr || realPath == nullptr || realPathLen < MMPA_MAX_PATH) {
    return EN_INVALID_PARAM;
  }

  std::string str_path = path;
  if (str_path.find("libcce.so") != std::string::npos) {
    strncpy(realPath, path, realPathLen);
    return EN_OK;
  }

  if (g_program_exiting.load(std::memory_order_relaxed)) {
      return EN_ERROR;
    }
    auto weak_impl = fe::MmpaStub::GetInstance().GetImplWeakPtr();
    auto impl = weak_impl.lock();
    if (!impl) {
      return EN_ERROR;
    }
    return impl->RealPath(path, realPath, realPathLen);
}

INT32 mmRWLockInit(mmRWLock_t *rwLock)
{
  if (rwLock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_init(rwLock, NULL);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockRDLock(mmRWLock_t *rwLock)
{
  if (rwLock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_rdlock(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockWRLock(mmRWLock_t *rwLock)
{
  if (rwLock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_wrlock(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRDLockUnLock(mmRWLock_t *rwLock)
{
  if (rwLock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_unlock(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmWRLockUnLock(mmRWLock_t *rwLock)
{
  if (rwLock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_unlock(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockDestroy(mmRWLock_t *rwLock)
{
  if (rwLock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_destroy(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmGetErrorCode()
{
  return 0;
}

INT32 mmIsDir(const CHAR *fileName)
{
  if (fileName == nullptr) {
    return EN_ERR;
  }

  DIR *pDir = opendir (fileName);
  if (pDir != nullptr) {
    (void) closedir (pDir);
    return EN_OK;
  }
  return EN_ERR;
}

INT32 mmGetEnv(const CHAR *name, CHAR *value, UINT32 len)
{
  const char *env = getenv(name);
  if (env == nullptr) {
    return EN_ERROR;
  }

  strncpy(value, env, len);
  return EN_OK;
}

CHAR *mmDlerror() {
  return dlerror();
}

INT32 mmDladdr(VOID *addr, mmDlInfo *info) {
  if (g_program_exiting.load(std::memory_order_relaxed)) {
      return -1;
  }
  auto weak_impl = fe::MmpaStub::GetInstance().GetImplWeakPtr();
  auto impl = weak_impl.lock();
  if (!impl) {
    return -1;
  }
  if (impl->DlAddr(addr, info) != -1) {
    return 0;
  } else {
    return -1;
  }
}

const static std::string libcce_name("libcce.so");
ge::ccStatus_t ccUpdateKernelArgs(ge::ccOpContext &, uint64_t, uint64_t, uint64_t, void *, uint64_t, void *) {
  return ge::ccStatus_t::CC_STATUS_SUCCESS;
}

VOID *mmDlopen(const CHAR *fileName, INT32 mode) {
  const std::string so_name = fileName;
  if (so_name.find("libcce.so") != std::string::npos) {
    return (void *)(libcce_name.data());
  }

  return fe::MmpaStub::GetInstance().GetImpl()->DlOpen(fileName, mode);
}

INT32 mmDlclose(VOID *handle) {
  if (libcce_name.data() == handle) {
    return 0;
  }
  if (handle == nullptr){
    return 1;
  }
  if (handle == (void *)0x8888) {
    return 0;
  }

  // 首先检查退出标志
  if (g_program_exiting.load(std::memory_order_relaxed)) {
    return 0; // 程序正在退出，安全跳过
  }
  auto weak_impl = fe::MmpaStub::GetInstance().GetImplWeakPtr();
  auto impl = weak_impl.lock();
  if (!impl) {
    return 0;
  }
  return impl->DlClose(handle);
}

VOID *mmDlsym(VOID *handle, const CHAR *funcName) {
  if (g_program_exiting.load(std::memory_order_relaxed)) {
 	     return nullptr;
  }
  auto weak_impl = fe::MmpaStub::GetInstance().GetImplWeakPtr();
  auto impl = weak_impl.lock();
  if (!impl) {
    return nullptr;
  }
  return impl->DlSym(handle, funcName);
}

INT32 mmGetPid()
{
  return (INT32)getpid();
}

INT32 mmSetCurrentThreadName(const CHAR *name)
{
  return EN_OK;
}

INT32 mmGetCwd(CHAR *buffer, INT32 maxLen)
{
  return EN_OK;
}

CHAR *mmGetErrorFormatMessage(mmErrorMSg errnum, CHAR *buf, mmSize size)
{
  if ((buf == NULL) || (size <= 0)) {
    return NULL;
  }
  return strerror_r(errnum, buf, size);
}

INT32 mmCreateTask(mmThread *threadHandle, mmUserBlock_t *funcBlock) {
  if ((threadHandle == NULL) || (funcBlock == NULL) || (funcBlock->procFunc == NULL)) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_create(threadHandle, NULL, funcBlock->procFunc, funcBlock->pulArg);
  if (ret != EN_OK) {
    ret = EN_ERROR;
  }
  return ret;
}

INT32 mmJoinTask(mmThread *threadHandle) {
  if (threadHandle == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_join(*threadHandle, NULL);
  if (ret != EN_OK) {
    ret = EN_ERROR;
  }
  return ret;
}

INT32 mmSleep(UINT32 millSecond) {
  if (millSecond == MMPA_ZERO) {
    return EN_INVALID_PARAM;
  }
  UINT32 microSecond;

  if (millSecond <= MMPA_MAX_SLEEP_MILLSECOND_USING_USLEEP) {
    microSecond = millSecond * (UINT32)MMPA_MSEC_TO_USEC;
  } else {
    microSecond = MMPA_MAX_SLEEP_MICROSECOND_USING_USLEEP;
  }

  INT32 ret = usleep(microSecond);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmUnlink(const CHAR *filename) {
  if (filename == NULL) {
    return EN_INVALID_PARAM;
  }
  return unlink(filename);
}

INT32 mmSetEnv(const CHAR *name, const CHAR *value, INT32 overwrite) {
  if ((name == nullptr) || (value == nullptr)) {
    return EN_INVALID_PARAM;
  }
  return setenv(name, value, overwrite);
}

INT32 mmWaitPid(mmProcess pid, INT32 *status, INT32 options) {
  if (g_program_exiting.load(std::memory_order_relaxed)) {
 	  return EN_ERROR;
  }
  auto weak_impl = fe::MmpaStub::GetInstance().GetImplWeakPtr();
  auto impl = weak_impl.lock();
  if (!impl) {
    return EN_ERROR;
  }
  return impl->WaitPid(pid, status, options);
}
#ifdef __cplusplus
}
#endif

namespace fe {
std::string GetCurpath() {
    Dl_info dl_info;
    if (dladdr((void*) GetCurpath, &dl_info) == 0) {
        return "";
    } else {
        std::string so_path = dl_info.dli_fname;
        char resoved_path[4096] = {0x00};
        realpath(so_path.c_str(), resoved_path);
        so_path = resoved_path;
        std::string real_dir_file_path = so_path.substr(0, so_path.rfind('/') + 1);
        return real_dir_file_path;
    }
}
}
