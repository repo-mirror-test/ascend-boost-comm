/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * MindKernelInfra is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>
#include "mki/tensor.h"
#include "../schedule/common/tensor.cpp"
namespace{
TEST(CalNumelTest, HandleValidInput){
	Mki::SVector<int64_t> v = {2, 3, 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 24);

	v = {1, 2, 3};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 6);

	v = {1000, 1000, 1000};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 1000000000);
}

TEST(CalNumelTest, HandleInvalidInput){
	Mki::SVector<int64_t> v = {-1, 3, 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 0);

	v = {2, 0, 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 0);

	v = {2, std::numeric_limits<int64_t>::max(), 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 3), 0);
}

TEST(CalNumelTest, HandleSubrangeInput){
	Mki::SVector<int64_t> v = {2, 3, 4};
	EXPECT_EQ(Mki::CalNumel(v, 0, 2), 6);

	v = {2, 3, 4};
	EXPECT_EQ(Mki::CalNumel(v, 1, 3), 12);
}

TEST(ViewTest, HandleValidInput) {
	Mki::TensorDesc td;
	td.dims = {3, 2, 4};
	Mki::SVector<int64_t> ms = {2, 3, 4};
	td.View(ms);
	EXPECT_EQ(td.dims, ms);

	Mki::SVector<int64_t> ms2 = {1, 2, 3};
	td.View(ms2);
	EXPECT_NE(td.dims, ms2);
}

TEST(ViewTest, HandleInvalidInput) {
	Mki::TensorDesc td;
	Mki::SVector<int64_t> ms = {2, 3, 4};
	td.View(ms);
	EXPECT_TRUE(td.dims.empty());
}

TEST(ToStringTest, HandleEmptyDims){
	Mki::TensorDesc td;
	td.dims = {};
	EXPECT_EQ(td.ToString(), "dtype:undefined, format:undefined, dims:[]");
}

TEST(ToStringTest, HandleSingleElementDims){
	Mki::TensorDesc td;
	td.dims = {10};
	EXPECT_EQ(td.ToString(), "dtype:undefined, format:undefined, dims:[10]");
}

TEST(ToStringTest, HandleMultipleElementsDims){
	Mki::TensorDesc td;
	td.dims = {1, 2, 3};
	EXPECT_EQ(td.ToString(), "dtype:undefined, format:undefined, dims:[1, 2, 3]");
}
} //namespace
