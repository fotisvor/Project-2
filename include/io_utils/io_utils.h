#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cmd_args.h"

using namespace std;
std::string USER_PromtExit(const std::string &);
std::string user_prompt_file(const std::string &);
size_t user_prompt_query_index(const std::string &, long, long);
uint32_t BigEND_LittleEND(uint32_t);

bool FileExists(const char *);
void LSH_parse(int, char *const[], LSH_Args **);
void Cube_Parse(int, char *const[], CubeArgs **);
void LSH_Usage(const char *);
void CubeUsage(const char *);
void user_interface(CubeArgs **);
void user_interface(LSH_Args **);
void PrintStats(const uint16_t, const size_t, const std::vector<std::vector<std::pair<uint32_t, size_t>>> &,
        const std::vector<std::chrono::microseconds> &, const std::vector<std::vector<uint32_t>> &,
        const std::vector<std::chrono::microseconds> &);

void write_output(const std::string &, const uint16_t, const size_t,
        const std::vector<std::vector<std::pair<uint32_t, size_t>>> &,
        const std::vector<std::chrono::microseconds> &, const std::vector<std::vector<uint32_t>> &,
        const std::vector<std::chrono::microseconds> &, const std::vector<std::vector<size_t>> &,
        const std::string &);

template <typename T> void read_file(const std::string &filepath, std::vector<std::vector<T>> &data)
{
    std::ifstream dataFILE;
    dataFILE.open(filepath, std::ios::in | std::ios::binary);

    if (dataFILE)
    {
        uint32_t MagicNumber, IMGnumber, ROWS, COLS;
        dataFILE.read((char *)&MagicNumber, sizeof(MagicNumber));
        MagicNumber = BigEND_LittleEND(MagicNumber);
        dataFILE.read((char *)&IMGnumber, sizeof(IMGnumber));
        IMGnumber = BigEND_LittleEND(IMGnumber);
        dataFILE.read((char *)&ROWS, sizeof(ROWS));
        ROWS = BigEND_LittleEND(ROWS);
        // read number of columns
        dataFILE.read((char *)&COLS, sizeof(COLS));
        COLS = BigEND_LittleEND(COLS);
        data.resize(IMGnumber, std::vector<T>(ROWS * COLS, 0));
        for (size_t i = 0; i != IMGnumber; ++i)
        {
            std::vector<T> &vec = data[i];

            for (size_t j = 0; j != ROWS * COLS; ++j)
            {
                dataFILE.read((char *)&vec[j], sizeof(vec[j]));
            }
        }

        dataFILE.close();
    }
    else
    {
        std::cerr << "Couldn't open the file!" << std::endl;
        exit(EXIT_FAILURE);
    }
}
