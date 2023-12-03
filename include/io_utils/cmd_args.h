#ifndef CMD_ARGS_H
#define CMD_ARGS_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class ProgArgs
{
  private:
    const std::string InputPath;
    std::string QueryPath;
    const std::string OUTputPath;
    uint16_t NNNumber = 0;
    float RADIUS = 0.0;

  public:
    ProgArgs(const std::string &, const std::string &, const std::string &, uint16_t, float);
    virtual ~ProgArgs() = default;
    void set_QueryPath(const std::string &);
    void Set_NNnumber(uint16_t);
    void set_RADIUS(float);
    std::string get_InputPath() const;
    std::string get_QueryPath() const;
    std::string get_OutputPath() const;
    uint16_t get_NearNeighborNum() const;
    float get_RADIUS() const;
    virtual uint32_t get_K() const = 0; 
};

class LSH_Args : public ProgArgs
{
  private:
    uint32_t HASHfunctionNum = 0; 
    uint16_t HASHtableNum = 0;    
  public:
    LSH_Args(const std::string &, std::string &, std::string &, uint16_t, float, uint32_t, uint16_t);
    LSH_Args(const std::string &, const std::string &, const std::string &);
    void set_HASHfunctionNum(uint16_t);
    void set_HashTableNum(uint16_t);
    uint32_t get_K() const;
    uint16_t get_HashTableNum() const;
};

class CubeArgs : public ProgArgs
{
  private:
    const uint32_t ProjectionD = 0; 
    const uint16_t MAX_cands = 0;   
    const uint16_t MAXprobes = 0;  

  public:
    CubeArgs(const std::string &, const std::string &, const std::string &, uint16_t, float, uint32_t, uint16_t,
             uint16_t);
    CubeArgs(const std::string &, const std::string &, const std::string &);
    uint32_t get_K() const;
    uint16_t get_MAXcandidates() const;
    uint16_t get_MAXprobes() const;
};

inline ProgArgs::ProgArgs::ProgArgs(const std::string &ipath, const std::string &qpath, const std::string &opath,
                                    uint16_t nn_num, float rad)
    : InputPath(ipath), QueryPath(qpath), OUTputPath(opath), NNNumber(nn_num), RADIUS(rad)
{
}

inline void ProgArgs::set_QueryPath(const std::string &qfile)
{
    QueryPath = qfile;
}

inline void ProgArgs::Set_NNnumber(uint16_t nns)
{
    NNNumber = nns;
}

inline void ProgArgs::set_RADIUS(float rad)
{
    RADIUS = rad;
}

inline std::string ProgArgs::get_InputPath() const
{
    return InputPath;
}

inline std::string ProgArgs::get_QueryPath() const
{
    return QueryPath;
}

inline std::string ProgArgs::get_OutputPath() const
{
    return OUTputPath;
}

inline uint16_t ProgArgs::get_NearNeighborNum() const
{
    return NNNumber;
}

inline float ProgArgs::get_RADIUS() const
{
    return RADIUS;
}





inline LSH_Args::LSH_Args(const std::string &ipath, std::string &qpath, std::string &opath, uint16_t nn_num, float rad,
                          uint32_t HashF_Num, uint16_t HashT_Num)
    : ProgArgs(ipath, qpath, opath, nn_num, rad), HASHfunctionNum(HashF_Num), HASHtableNum(HashT_Num)
{
}

inline LSH_Args::LSH_Args(const std::string &ipath, const std::string &qpath, const std::string &opath)
    : ProgArgs(ipath, qpath, opath, 1, 10000.0), HASHfunctionNum(4), HASHtableNum(5)

{
}


inline void LSH_Args::set_HASHfunctionNum(uint16_t HashF_Num)
{
    HASHfunctionNum = HashF_Num;
}

inline void LSH_Args::set_HashTableNum(uint16_t HashT_Num)
{
    HASHtableNum = HashT_Num;
}

inline uint32_t LSH_Args::get_K() const
{
    return HASHfunctionNum;
}

inline uint16_t LSH_Args::get_HashTableNum() const
{
    return HASHtableNum;
}



inline CubeArgs::CubeArgs(const std::string &ipath, const std::string &qpath, const std::string &opath, uint16_t nn_num,
                          float rad, uint32_t proj, uint16_t max_cands, uint16_t MAXprobes)
    : ProgArgs(ipath, qpath, opath, nn_num, rad), ProjectionD(proj), MAX_cands(max_cands), MAXprobes(MAXprobes)
{
}

inline CubeArgs::CubeArgs(const std::string &ipath, const std::string &qpath, const std::string &opath)
    : ProgArgs(ipath, qpath, opath, 1, 10000.0), ProjectionD(14), MAX_cands(10), MAXprobes(2)
{
}

inline uint32_t CubeArgs::get_K() const
{
    return ProjectionD;
}

inline uint16_t CubeArgs::get_MAXcandidates() const
{
    return MAX_cands;
}

inline uint16_t CubeArgs::get_MAXprobes() const
{
    return MAXprobes;
}

#endif
