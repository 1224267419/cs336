# Spring 2026 CS336 lectures

This repository contains the lecture materials for Stanford's Language Modeling from Scratch (CS336).

## Executable lectures

These are named `lecture_XX.py`.

### Setup

        uv sync
        git clone https://github.com/percyliang/edtrace

You can compile a lecture by running:

        python execute.py -m lecture_01

which generates a `var/traces/lecture_01.json` and caches any images as
appropriate.

To view it locally:

Load a local server to view at `http://localhost:5173?trace=var/traces/sample.json`:

        npm run --prefix=edtrace/frontend dev

Deploy to the main website:

        npm run --prefix=edtrace/frontend build
        git add assets
        git ci -am "<some message>"
        git push

## Non-executable lectures

These are named `lecture_XX.pdf`.

##### scaling laws

由于规模增长等问题, 在一定预算的算力规模下只能完成一个任务 , 如何根据能够计算的确定的memory和flops 来推测模型大小以及结构是需要考虑的

scaling law 并非自动出现 ,而是需要精心构造超参数实现

predictability is at least as important as optimality 可预测性至少与最优性一样重要, 这样才能便于我们讲参数化的结论外推

经典的scaling law中given a FLOPs budget(C=6 N D) , 该选择use a bigger model(N)or train on more tokens(D)?
