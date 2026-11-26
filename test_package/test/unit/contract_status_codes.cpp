// Conan::ImportStart
#include <gtest/gtest.h>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*
 * `infr_status_str` is public API that nothing tested and nothing called, which is
 * exactly the shape of a function that drifts: the spelling of a status is what a
 * log line or a diagnostic prints, and a rename would change what every consumer's
 * error message says without a single test noticing.
 *
 * The numeric values are the other half of the contract: the header states they are
 * stable across releases, because a status crossing an ABI boundary is a number
 * before it is a name. Pinning them here is what makes that statement checkable.
 *
 * `infr_status_str` 是**公开 API**，此前既无测试也无调用方——这正是容易漂移的形状：
 * 状态的拼写就是日志与诊断打印出来的东西，改一次名字，所有消费者的错误信息都变了，
 * 而没有一条测试会察觉。数值是契约的另一半：头文件声明它们跨版本稳定，因为跨 ABI 边界的
 * 状态先是数字、然后才是名字。把数值钉在这里，那句声明才可被检查。
 */

TEST(ContractStatusCodes, NumericValuesAreStableAcrossReleases) {
    EXPECT_EQ(0, static_cast<int>(INFR_OK));
    EXPECT_EQ(1, static_cast<int>(INFR_ERR_INVALID_ARGUMENT));
    EXPECT_EQ(2, static_cast<int>(INFR_ERR_UNSUPPORTED));
    EXPECT_EQ(3, static_cast<int>(INFR_ERR_BUFFER_TOO_SMALL));
    EXPECT_EQ(4, static_cast<int>(INFR_ERR_NOT_IMPLEMENTED));
}

TEST(ContractStatusCodes, EveryCodeHasItsOwnSpelling) {
    EXPECT_STREQ("INFR_OK", infr_status_str(INFR_OK));
    EXPECT_STREQ("INFR_ERR_INVALID_ARGUMENT", infr_status_str(INFR_ERR_INVALID_ARGUMENT));
    EXPECT_STREQ("INFR_ERR_UNSUPPORTED", infr_status_str(INFR_ERR_UNSUPPORTED));
    EXPECT_STREQ("INFR_ERR_BUFFER_TOO_SMALL", infr_status_str(INFR_ERR_BUFFER_TOO_SMALL));
    EXPECT_STREQ("INFR_ERR_NOT_IMPLEMENTED", infr_status_str(INFR_ERR_NOT_IMPLEMENTED));
}

TEST(ContractStatusCodes, AnUndeclaredCodeIsNamedRatherThanGuessed) {
    // A value from a newer library, or a caller that stored an int in the wrong
    // place: the answer must be visibly "unknown", never a plausible neighbour.
    EXPECT_STREQ("INFR_STATUS_UNKNOWN", infr_status_str(static_cast<infr_status>(99)));
    EXPECT_STREQ("INFR_STATUS_UNKNOWN", infr_status_str(static_cast<infr_status>(-1)));
}
