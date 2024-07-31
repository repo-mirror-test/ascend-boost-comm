/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>
#include "mki/utils/log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "device_version_check.h"
#include "platform/platform_infos_def.h"
#include "mki/utils/platform/platform_manager.h"
#include "utils/platform/platform_infos_impl.h"

TEST(PlatformTest, platformTest1)
{
    const uint32_t PLATFORM_SUCCESS = 0;
    Mki::PlatformManager &platformManager = Mki::PlatformManager::Instance();
    uint32_t ret = platformManager.InitializePlatformManager();
    EXPECT_EQ(ret, PLATFORM_SUCCESS);
    fe::PlatFormInfos platformInfo;
    ret = platformManager.GetPlatformInfos("Ascend910A", platformInfo);
    EXPECT_EQ(ret, PLATFORM_SUCCESS);
    std::string val;
    platformInfo.GetPlatformRes("SoCInfo", "ai_core_cnt", val);
    EXPECT_EQ(val, "32");
}

TEST(PlatformInfoTest, SupportL0c2out) {
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::PlatformInfo& platformInfo = Mki::PlatformInfo::PlatformInfo::Instance();
    bool result = platformInfo.SupportL0c2out();
    ASSERT_TRUE(result);
}

TEST(PlatformInfoTest, SupportL0c2out0) {
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::string label = "platform1";
    uint64_t size = 10;
    uint64_t size1 = 0;
    std::map<std::string, std::string> res;
    platformInfo.GetPlatformRes(label, res);
    platformInfo.GetLocalMemSize(fe::LocalMemType::HBM, size);
    platformInfo.GetLocalMemSize(fe::LocalMemType::HBM, size1);
    platformInfo.GetLocalMemBw(fe::LocalMemType::L2, size1);
    platformInfo.GetLocalMemBw(fe::LocalMemType::HBM,size1);
    platformInfo.GetLocalMemBw(fe::LocalMemType::L2, size);
    platformInfo.GetFixPipeDtypeMap();
}

TEST(PlatFormInfosImplTest, GetPlatformRes){
    std::string label = "platform1";
    fe::PlatFormInfosImpl platformInfoimp;
    std::map<std::string, std::string> res;
    ASSERT_FALSE(platformInfoimp.GetPlatformRes(label, res));
    std::map<std::string, std::vector<std::string>> map1;
    EXPECT_EQ(platformInfoimp.GetFixPipeDtypeMap(), map1);
}
TEST(PlatformManagerTest, Finalize){
    Mki::PlatformManager &platformManager = Mki::PlatformManager::Instance();
    EXPECT_EQ(platformManager.Finalize(), 0);
}