// #include "setdiff.h"

// #include <catch2/catch.hpp>

// TEST_CASE("set_difference()")
// {
//     // add tests here
// }

// 1.首先宏定函数名字
// 2.auto是自动定义函数类型,定义空数组
// 3.定义bl为char类型,包含元素o和k
// 4.定义期待输出
// 5.引用外部函数移除vs中存在bl的元素
// 6.检查是否为期待输出结果

// TEST_CASE("Empty vector unchanged"){
//     auto vs = std::vector<char>{};
//     auto blacklist = std::vector<char>{'o','k'};
//     const auto expected = std::vector<char>{};
//     set_difference = (vs,blacklist);
//     CHECK(vs == expected);
// }

#ifndef COMP6771_SETDIFF_H
#define COMP6771_SETDIFF_H

#include <vector>

/**
 * Removes all occurrences of each element in blacklist from vec_set
 * @param vec_set The vector set who will have its elements removed
 * @param blacklist The list of elements to remove
 */
auto set_difference(std::vector<char>& vec_set, const std::vector<char>& blacklist) -> void;

#endif // COMP6771_SETDIFF_H

#include "setdiff.h"

#include <catch2/catch.hpp>

TEST_CASE("set_difference()")
{
}
// #include "setdiff.h"
// #include <catch2/catch.hpp>
#include <vector>

TEST_CASE("set_difference - Handles Invalid Input") {
    std::vector<char> vec_set = {'a', 'b', 'c', 'd'};
    std::vector<char> blacklist = {'\0', '\n', ' '}; // invail elements

    set_difference(vec_set, blacklist);
    
    // ignore invaild elements ,should be orgin ones.
    std::vector<char> expected = {'a', 'b', 'c', 'd'};
    REQUIRE(vec_set == expected);
}

TEST_CASE("set_difference - Handles Edge Case") {
    std::vector<char> vec_set = {};
    std::vector<char> blacklist = {'a', 'b', 'c'};

    set_difference(vec_set, blacklist);
    
    // should be empty vector
    std::vector<char> expected = {};
    REQUIRE(vec_set == expected);
}

TEST_CASE("set_difference - Handles Average Case") {
    std::vector<char> vec_set = {'a', 'b', 'c', 'd', 'e'};
    std::vector<char> blacklist = {'b', 'd'};

    set_difference(vec_set, blacklist);
    
    // should be remove vector in the blacklist
    std::vector<char> expected = {'a', 'c', 'e'};
    REQUIRE(vec_set == expected);
}

TEST_CASE("set_difference - Handles No Common Elements") {
    std::vector<char> vec_set = {'a', 'b', 'c'};
    std::vector<char> blacklist = {'x', 'y', 'z'};

    set_difference(vec_set, blacklist);
    
    // should be orgin vector
    std::vector<char> expected = {'a', 'b', 'c'};
    REQUIRE(vec_set == expected);
}

TEST_CASE("set_difference - Handles All Elements to Remove") {
    std::vector<char> vec_set = {'a', 'b', 'c'};
    std::vector<char> blacklist = {'a', 'b', 'c'};

    set_difference(vec_set, blacklist);
    
    // should be empty vector
    std::vector<char> expected = {};
    REQUIRE(vec_set == expected);
}
}