/*!
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#include <mxnet/base.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "./storage_manager.h"

namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager with a memory pool.
 */
template <class DeviceStorage, size_t kThreshold>
class PooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  PooledStorageManager() = default;
  /*!
   * \brief Default destructor.
   */
  ~PooledStorageManager() {
    ReleaseAll();
  }
  void* Alloc(size_t size) override;
  void Free(void* ptr, size_t size) override;

 private:
  void ReleaseAll();
  // internal mutex
  std::mutex mutex_;
  // used memory
  size_t used_memory_ = 0;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(PooledStorageManager);
};  // class PooledStorageManager

template <class DeviceStorage, size_t kThreshold>
void* PooledStorageManager<DeviceStorage, kThreshold>::Alloc(size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto&& reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
    if (kThreshold <= used_memory_) {
      ReleaseAll();
    }
    used_memory_ += size;
    return DeviceStorage::Alloc(size);
  } else {
    auto&& reuse_pool = reuse_it->second;
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    return ret;
  }
}

template <class DeviceStorage, size_t kThreshold>
void PooledStorageManager<DeviceStorage, kThreshold>::Free(void* ptr,
                                                           size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto&& reuse_pool = memory_pool_[size];
  reuse_pool.push_back(ptr);
}

template <class DeviceStorage, size_t kThreshold>
void PooledStorageManager<DeviceStorage, kThreshold>::ReleaseAll() {
  for (auto&& i : memory_pool_) {
    for (auto&& j : i.second) {
      DeviceStorage::Free(j);
      used_memory_ -= i.first;
    }
  }
  memory_pool_.clear();
}

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
