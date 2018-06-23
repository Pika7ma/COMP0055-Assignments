#include <stdio.h>
#include <immintrin.h>
#include <algorithm>
#include <limits.h>
#include <Windows.h>

const int MAX_N = 4096;
const int KN_SIZE = 128;
const int NUM_THREADS = 4;
float m[MAX_N][MAX_N];
float m_tmp[MAX_N][MAX_N - KN_SIZE + 1];
float m_[MAX_N - KN_SIZE + 1][MAX_N - KN_SIZE + 1];

void reset(const int n, const int kn_size, float m[][MAX_N], float m_[][MAX_N - KN_SIZE + 1]) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            m[i][j] = (float)rand();
        }
    }
}

void max_pooling(const int n, const int kn_size, float m[][MAX_N], float m_[][MAX_N - KN_SIZE + 1]) {
    for (int i = 0; i < n - kn_size + 1; ++i) {
        for (int j = 0; j < n - kn_size + 1; ++j) {
            float max_ = -FLT_MAX;
            for (int k = 0; k < kn_size; ++k) {
                for (int l = 0; l < kn_size; ++l) {
                    if (m[i + k][j + l] > max_) {
                        max_ = m[i + k][j + l];
                    }
                }
            }
            m_[i][j] = max_;
        }
    }
}

// Our Method
void max_pooling_(const int n, const int kn_size, float m[][MAX_N], float m_[][MAX_N - KN_SIZE + 1]) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - kn_size + 1; ++j) {
            float max_ = -FLT_MAX;
            for (int k = 0; k < kn_size; ++k) {
                if (m[i][j + k] > max_) {
                    max_ = m[i][j + k];
                }
            }
            m_tmp[i][j] = max_;
        }
    }
    for (int i = 0; i < n - kn_size + 1; ++i) {
        for (int j = 0; j < n - kn_size + 1; ++j) {
            float max_ = -FLT_MAX;
            for (int k = 0; k < kn_size; ++k) {
                if (m_tmp[i + k][j] > max_) {
                    max_ = m[i + k][j];
                }
            }
            m_[i][j] = max_;
        }
    }
}


void print_matrix(const int n, float m[][MAX_N]) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%6.1f ", m[i][j]);
        }
        printf("\n");
    }
}


int main(int argc, char* argv[]) {
    // set random seed
    srand(0);
    reset(MAX_N, KN_SIZE, m, m_);

    // create counter var
    long long head, tail, freq;

    // init frequency
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // begin time counter
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    reset(MAX_N, KN_SIZE, m, m_);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    double t_reset = (tail - head) * 1.0 / freq;

    ////////////////////////////////////////////////
    // begin time counter
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    reset(MAX_N, KN_SIZE, m, m_);
    max_pooling_(MAX_N, KN_SIZE, m, m_);

    // stop time counter
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);

    // verbose
    double t_max_pooling = (tail - head) * 1.0 / freq;
    printf("time: %lf\n", t_max_pooling - t_reset);
    ////////////////////////////////////////////////

    return 0;
}
