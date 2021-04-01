---
title: leetcode 295 Find Median from Data Stream
date: 2021-03-29 19:45:06
categories: 题解
tags:
- 最大堆
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/find-median-from-data-stream/)

The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

- For example, for `arr = [2,3,4]`, the median is `3`.
- For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.

Implement the MedianFinder class:

- `MedianFinder()` initializes the `MedianFinder` object.
- `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
- `double findMedian()` returns the median of all elements so far. Answers within `10-5` of the actual answer will be accepted.

 

**Example 1:**

```
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

 

**Constraints:**

- `-105 <= num <= 105`
- There will be at least one element in the data structure before calling `findMedian`.
- At most `5 * 104` calls will be made to `addNum` and `findMedian`.

 

**Follow up:**

- If all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?
- If `99%` of all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?

## 方法1 最大堆和最小堆

### 思路

数据是动态并且保持有序，所以很容易想到使用红黑树，也就是使用优先队列来存储数据。但是如何将获得中位数的复杂度降低才是本题的难点，如果使用topk的思路每次需要出队和入队，也就是O(N)的复杂度，所以只用一颗红黑树是不够的。

将数据分为左右两部分，记录左边的最大值p1，记录右边的最小值p2,则保证左右两边的数据量一样大，就可以用O(1)的时间得到中位数$(p1+p2)/2$（总数为偶数的情况）。

因此可以使用两颗红黑树分别存储数据，其中左边使用最大堆记录最大值，右边使用最小堆记录最小值。但是需要保证

1. 左右两边的个数相差不超过1个。
2. 左边存储的元素一定要比右边的元素小。

第一个可以用当前数据总数为偶数时插入到左边，否则插入到右边来保证。也就是轮流左右插入。

第二个则有两种特殊情况

1. 插入左边的元素比右边的元素大
2. 插入右边的元素比左边的元素小

针对第一种情况进行讲解：

直接插入左边是肯定不行，因此先插入右边的最小堆，然后将更新之后的最小堆的最小值pop出来插入到左边。因为pop出来的值肯定小于右边的数据，所以插入到左边就能保证顺序了。

### 复杂度

插入的时间复杂度为红黑树的插入时间复杂度$O(Log(N))$

查询中位数的时间复杂度$O(1)$

### 代码

```cc
/*
Runtime: 92 ms, faster than 95.16% of C++ online submissions for Find Median from Data Stream.
Memory Usage: 46.8 MB, less than 33.04% of C++ online submissions for Find Median from Data Stream.
*/
class MedianFinder {
private:
    priority_queue<int, vector<int>, less<int>> leftHeap;//左边用最大堆
    priority_queue<int, vector<int>, greater<int>> rightHeap;//右边用最小堆
    int TotalNums; //record the total nums of left and right heap
public:
    /** initialize your data structure here. */
    MedianFinder() {
        leftHeap = {};
        rightHeap = {};
        TotalNums = 0;
    }
    
    void addNum(int num) {
        if (TotalNums % 2 == 0) {
            if (!rightHeap.empty() && rightHeap.top() < num) { // 左边插入一个比右边大得处理
                rightHeap.push(num);
                leftHeap.push(rightHeap.top());
                rightHeap.pop();
            }else {
                leftHeap.push(num);
            }            
        } else {
            if (leftHeap.top() > num) { // 右边插入一个比左边小得处理
                leftHeap.push(num);
                rightHeap.push(leftHeap.top());
                leftHeap.pop();
            }else{
                rightHeap.push(num);
            }
        }
        TotalNums++;
    }
    
    double findMedian() {
        double median;
        if (TotalNums % 2 == 0) {
            median = ((double) leftHeap.top() + rightHeap.top()) / 2;
        } else {
            median = leftHeap.top();
        }
        return median;
    }
};
```

## 总结

1. 同一个数据类型的重复，大小堆的写法（greater是最小堆，less是最大堆）
2. 时间复杂度的权衡

## 参考

《剑指offer》