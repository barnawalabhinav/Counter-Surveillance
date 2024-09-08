#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

#include "helper.hpp"

// Custom reduction declaration for OpenMP
#pragma omp declare reduction(+ : Stats : omp_out += omp_in) initializer(omp_priv = Stats())

/*
Simulate the decentralized attendance system in a class

# Parameters:
n: Number of Students in the Class
m: Number of Students to pick for rollcall
k: Minimum Number of Votes required to be present
a: Maximum Number of students that each student can ask for vote
p: Probability that a student is marked present if he/she is present
q: Probability that a student is marked present if he/she is absent
r: Probability that a student is present in the class
seed: Random Number Generator Seed
b: b for voting for absent student
*/
inline Stats simulate(int n, int m, int k, int a, float p, float q, float r, int seed = -1, int b = 5, bool printVoteCount = false)
{
    std::bernoulli_distribution vote_present_dist(p);
    std::bernoulli_distribution vote_absent_dist(q);
    std::bernoulli_distribution is_present_dist(r);

    std::random_device rd;
    std::mt19937 generator(rd());
    if (seed != -1)
        generator = std::mt19937(seed);

    std::vector<bool> present_status(n);
    std::vector<int> voters[n];
    std::vector<int> students(n);
    for (int i = 0; i < n; ++i)
    {
        students[i] = i;
        int is_present = is_present_dist(generator);
        if (is_present)
            present_status[i] = true;
        else
            present_status[i] = false;
    }

    // Gather votes from students
    for (int i = 0; i < n; i++)
    {
        std::shuffle(students.begin(), students.end(), generator);
        // int a_i = rand() % (a + 1);
        int a_i = a;
        for (int j = 0; j < a_i; j++)
        {
            if (students[j] == i)
            {
                a_i++;
                continue;
            }

            if (present_status[i])
            {
                if (vote_present_dist(generator))
                    voters[i].push_back(students[j]);
            }
            else
            {
                if (vote_absent_dist(generator))
                    voters[i].push_back(students[j]);
            }
        }
    }

    // Roll Call random students and check if they are present
    std::vector<int> attendance_criteria(n, k); // Effective k for each student separately
    std::vector<bool> marked_status(n), marked_present(n);
    std::shuffle(students.begin(), students.end(), generator);
    for (int i = 0; i < m; i++)
    {
        int student = students[i];
        marked_present[student] = present_status[student];
        marked_status[student] = true;
        if (!present_status[student])
        {
            attendance_criteria[student] = n + 1;
            for (int voter : voters[student])
                attendance_criteria[voter] += b;
        }
    }

    // Check if each student is present or not
    for (int i = 0; i < n; i++)
    {
        if (marked_status[i])
            continue;
        // std::cout << "Student " << i << " : " << voters[i].size() << std::endl;
        if (voters[i].size() >= attendance_criteria[i])
            marked_present[i] = true;
        else
            marked_present[i] = false;
    }

    Stats stats;
    stats.calc_stats(present_status, marked_present);

    if (printVoteCount)
    {
        double mean_k_cnt = 0;
        double std_k_cnt = 0;
        for (int i = 0; i < n; i++)
        {
            mean_k_cnt += voters[i].size();
            std_k_cnt += voters[i].size() * voters[i].size();
        }
        mean_k_cnt /= n;
        std_k_cnt /= n;
        std_k_cnt -= mean_k_cnt * mean_k_cnt;
        std_k_cnt = sqrt(std_k_cnt);

        std::cout << std::fixed << std::setprecision(3) << "Number of Voters : " << mean_k_cnt << "\t\u00B1\t" << std_k_cnt << std::endl;
    }
    return stats;
}

inline std::pair<Stats, Stats> experiment(int n, int m, int k, int a, float p, float q, float r, int seed = -1, int b = 5, int threads = 1)
{
    Stats res = simulate(n, m, k, a, p, q, r, seed, b, true);

    Stats mean, std;
    int num_sim = 0;
#pragma omp parallel for reduction(+ : mean, std, num_sim) num_threads(threads)
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            Stats result = simulate(n, m, k, a, p, q, r, seed, b);
            if (result.is_valid())
            {
                mean += result;
                std += result * result;
                num_sim++;
                break;
            }
        }
    }
    mean /= num_sim;
    std /= num_sim;
    std -= mean * mean;
    std = std.sqrt();
    return {mean, std};
}

void display_help(const std::string &program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --help          Display this help message\n"
              << "  --threads <int> Specify the Number of Parallel Threads to use\n"
              << "  --seed <int>    Specify the random number generator seed\n"
              << "  --sim <int>     Specify the Number of Simulations\n"
              << "  --n <int>       Specify the Number of Students in the Class\n"
              << "  --m <int>       Specify the Number of Students to pick for rollcall\n"
              << "  --k <int>       Specify the Minimum Number of Votes required to be present\n"
              << "  --a <int>       Specify the Maximum Number of students that each student can ask for vote\n"
              << "  --b <int>       Specify the Penalty for voting for absent student\n"
              << "  --p <float>     Specify the Probability that a student is marked present if he/she is present\n"
              << "  --q <float>     Specify the Probability that a student is marked present if he/she is absent\n"
              << "  --r <float>     Specify the Probability that a student is present in the class\n"
              << std::endl;
}

int main(int argc, char *argv[])
{
    // ################## PARSING CMD LINE ARGS ##################
    int num_threads = 1; // Parallel Threads Count
    int seed = -1;       // Random Number Generator Seed
    int sim = 1000;      // Simmulations
    int n = 100;         // Strength
    int m = 10;          // Roll Call Count
    int k = 5;           // Attendance Criteria
    int a = 5;           // Vote Limit
    int b = 5;           // Penalty for voting for absent student
    float p = 0.9;       // Vote Present Prob
    float q = 0.1;       // Vote Absent Probt
    float r = 0.6;       // Present Prob

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {
            display_help(argv[0]);
            return 0;
        }
        else if (arg == "--threads" && i + 1 < argc)
            num_threads = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc)
            seed = std::stoi(argv[++i]);
        else if (arg == "--sim" && i + 1 < argc)
            sim = std::stoi(argv[++i]);
        else if (arg == "--n" && i + 1 < argc)
            n = std::stoi(argv[++i]);
        else if (arg == "--m" && i + 1 < argc)
            m = std::stoi(argv[++i]);
        else if (arg == "--k" && i + 1 < argc)
            k = std::stoi(argv[++i]);
        else if (arg == "--a" && i + 1 < argc)
            a = std::stoi(argv[++i]);
        else if (arg == "--b" && i + 1 < argc)
            b = std::stoi(argv[++i]);
        else if (arg == "--p" && i + 1 < argc)
            p = std::stof(argv[++i]);
        else if (arg == "--q" && i + 1 < argc)
            q = std::stof(argv[++i]);
        else if (arg == "--r" && i + 1 < argc)
            r = std::stof(argv[++i]);
        else
        {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            display_help(argv[0]);
            return 1;
        }
    }

    // ################## VALIDATING INPUTS ##################
    if (n < 1)
    {
        std::cerr << "Error: Number of Students in the Class should be greater than 0" << std::endl;
        return 1;
    }
    if (m < 0)
    {
        std::cerr << "Error: Number of Students to pick for rollcall should be greater than or equal to 0" << std::endl;
        return 1;
    }
    if (k < 0)
    {
        std::cerr << "Error: Minimum Number of Votes required to be present should be greater than or equal to 0" << std::endl;
        return 1;
    }
    if (a < 0)
    {
        std::cerr << "Error: Maximum Number of students that each student can ask for vote should be greater than or equal to 0" << std::endl;
        return 1;
    }
    if (b < 0)
    {
        std::cerr << "Error: Penalty for voting for absent student should be greater than or equal to 0" << std::endl;
        return 1;
    }
    if (p < 0 || p > 1)
    {
        std::cerr << "Error: Probability that a student is marked present if he/she is present should be between 0 and 1" << std::endl;
        return 1;
    }
    if (q < 0 || q > 1)
    {
        std::cerr << "Error: Probability that a student is marked present if he/she is absent should be between 0 and 1" << std::endl;
        return 1;
    }
    if (r < 0 || r > 1)
    {
        std::cerr << "Error: Probability that a student is present in the class should be between 0 and 1" << std::endl;
        return 1;
    }
    if (m > n)
    {
        std::cerr << "Error: Number of students to pick for rollcall should be less than or equal to the number of students in the class" << std::endl;
        return 1;
    }
    if (a > n - 1)
    {
        std::cerr << "Error: Maximum Number of students that each student can ask for vote should be less than the number of students in the class" << std::endl;
        return 1;
    }
    if (k > n - 1)
    {
        std::cerr << "Error: Minimum Number of Votes required to be present should be less than the number of students in the class" << std::endl;
        return 1;
    }

    // ################## PRINTING EXPERIMENT PARAMETERS ##################
    std::cout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
    std::cout << "Number of Threads -------------------------------------------- threads: " << num_threads << std::endl;
    std::cout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
    std::cout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
    std::cout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
    std::cout << "Number of Students to pick for rollcall ---------------------------- m: " << m << std::endl;
    std::cout << "Minimum Number of Votes required to be present --------------------- k: " << k << std::endl;
    std::cout << "Maximum Number of students that each student can ask for vote ------ a: " << a << std::endl;
    std::cout << "Penalty for voting a absent student -------------------------------- b: " << a << std::endl;
    std::cout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
    std::cout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
    std::cout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
    std::cout << "---------------------------------------------------------------------------- " << std::endl;

    std::cout << "############################ SIMULATION RESULTS ############################" << std::endl;
    std::pair<Stats, Stats> result = experiment(n, m, k, a, p, q, r, seed, b, num_threads);
    std::cout << result << std::endl;
    std::cout << "############################################################################ " << std::endl;

    return 0;
}
