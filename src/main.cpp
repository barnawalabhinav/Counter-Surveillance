#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

#include "helper.hpp"
#include "plotter.hpp"

// Custom reduction declaration for OpenMP
#pragma omp declare reduction(+ : Stats : omp_out += omp_in) initializer(omp_priv = Stats())

/**
* @brief Simulate the decentralized attendance system in a class

* @param n: Number of Students in the Class
* @param m: Number of Students to pick for rollcall
* @param k: Minimum Number of Votes required to be present
* @param a: Maximum Number of students that each student can ask for vote
* @param p: Probability that a student is marked present if he/she is present
* @param q: Probability that a student is marked present if he/she is absent
* @param r: Probability that a student is present in the class
* @param seed: Random Number Generator Seed
* @param b: b for voting for absent student
*/
inline Stats simulate(int n, float m, float k, float a, float p, float q, float r, int seed = -1, float b = 0.05, bool printVoteCount = false)
{
    m = std::round(n * m);
    a = std::round(n * a);
    k = std::round(n * k);
    b = std::round(n * b);

    std::bernoulli_distribution vote_present_dist(p);
    std::bernoulli_distribution vote_absent_dist(q);
    std::bernoulli_distribution is_present_dist(r);

    std::random_device rd;
    std::mt19937 generator(rd());
    if (seed != -1)
        generator = std::mt19937(seed);

    std::vector<bool> present_status(n);
    std::vector<std::vector<int>> voters(n, std::vector<int>());
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

inline std::pair<Stats, Stats> experiment(int n, float m, float k, float a, float p, float q, float r, int seed = -1, float b = 0.05)
{
    // Stats res = simulate(n, m, k, a, p, q, r, seed, b, true);

    Stats mean, std;
    int num_sim = 0;
    for (int i = 0; i < 1000; i++)
    {
        Stats result = simulate(n, m, k, a, p, q, r, seed, b);
        mean += result;
        std += result * result;
        num_sim++;
    }
    mean /= num_sim;
    std /= num_sim;
    std -= mean * mean;
    std = std.sqrt();
    return {mean, std};
}

// Function to compute the least squares coefficient for a given variable
template <typename T1, typename T2>
inline void fitLinear(const std::vector<T1> &x, std::vector<std::vector<T2>> &y, std::string var_name)
{
    std::string metric[5] = {"Accuracy", "Precision", "Recall", "F1-Score", "MCC"};

    for (int i = 0; i < y.size(); i++)
    {
        double sum_x = 0.0, sum_x2 = 0.0, sum_y = 0.0, sum_xy = 0.0;
        for (int j = 0; j < x.size(); j++)
        {
            sum_x += x[j];
            sum_x2 += x[j] * x[j];
            sum_y += y[i][j];
            sum_xy += x[j] * y[i][j];
        }
        double c = sum_xy / sum_x2;
        std::cout << var_name << " varies as (" << c << " * n) when optimizing for " << metric[i] << std::endl;
    }
}

void display_help(const std::string &program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --help          Display this help message\n"
              << "  --mode <str>    Specify the mode <plot or regress> \n"
              << "  --threads <int> Specify the Number of Parallel Threads to use\n"
              << "  --seed <int>    Specify the random number generator seed\n"
              << "  --sim <int>     Specify the Number of Simulations\n"
              << "  --n <int>       Specify the Number of Students in the Class\n"
              << "  --m <float>     Specify the Fraction of Students to pick for rollcall\n"
              << "  --k <float>     Specify the Minimum Number of Votes required to be present as fraction of n\n"
              << "  --a <float>     Specify the Fraction of students that each student can ask for vote\n"
              << "  --b <float>     Specify the Penalty for voting for absent student as fraction of n\n"
              << "  --p <float>     Specify the Probability that a student is marked present if he/she is present\n"
              << "  --q <float>     Specify the Probability that a student is marked present if he/she is absent\n"
              << "  --r <float>     Specify the Probability that a student is present in the class\n"
              << std::endl;
}

int main(int argc, char *argv[])
{
    // ################## PARSING CMD LINE ARGS ##################
    std::string mode = "plot"; // Mode of the program
    int threads = 1;           // Parallel Threads Count
    int seed = -1;             // Random Number Generator Seed
    int sim = 1000;            // Simmulations
    int n = 100;               // Strength
    float m = 0.1;             // Roll Call Count
    float k = 0.1;             // Attendance Criteria
    float a = 0.3;             // Vote Limit
    float b = 0.05;            // Penalty for voting for absent student
    float p = 0.75;            // Vote Present Prob
    float q = 0.25;            // Vote Absent Probt
    float r = 0.5;             // Present Prob

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {
            display_help(argv[0]);
            return 0;
        }
        else if (arg == "--mode" && i + 1 < argc)
            mode = argv[++i];
        else if (arg == "--threads" && i + 1 < argc)
            threads = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc)
            seed = std::stoi(argv[++i]);
        else if (arg == "--sim" && i + 1 < argc)
            sim = std::stoi(argv[++i]);
        else if (arg == "--n" && i + 1 < argc)
            n = std::stoi(argv[++i]);
        else if (arg == "--m" && i + 1 < argc)
            m = std::stof(argv[++i]);
        else if (arg == "--k" && i + 1 < argc)
            k = std::stof(argv[++i]);
        else if (arg == "--a" && i + 1 < argc)
            a = std::stof(argv[++i]);
        else if (arg == "--b" && i + 1 < argc)
            b = std::stof(argv[++i]);
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
    if (m > 1)
    {
        std::cerr << "Error: Number of students to pick for rollcall should be less than or equal to the number of students in the class" << std::endl;
        return 1;
    }
    if (a > 1)
    {
        std::cerr << "Error: Maximum Number of students that each student can ask for vote should be less than the number of students in the class" << std::endl;
        return 1;
    }
    if (k > a)
    {
        std::cerr << "Error: Minimum Number of Votes required to be present should be less than the number of students in the class" << std::endl;
        return 1;
    }

    // ################## PRINTING EXPERIMENT PARAMETERS ##################
    std::cout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
    std::cout << "Mode of the program --------------------------------------------- mode: " << mode << std::endl;
    std::cout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
    std::cout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
    std::cout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
    std::cout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
    std::cout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
    std::cout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
    std::cout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
    std::cout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
    std::cout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
    std::cout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
    std::cout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
    std::cout << "---------------------------------------------------------------------------- " << std::endl;

    int num_vars = 11;

    std::vector<float> x(num_vars);
    std::vector<double> ub(num_vars), lb(num_vars);
    if (mode == "plot")
    {
        std::vector<double> tp_mean(num_vars);
        std::vector<double> fp_mean(num_vars);
        std::vector<double> fn_mean(num_vars);
        std::vector<double> tn_mean(num_vars);
        std::vector<double> acc_mean(num_vars);
        std::vector<double> precision_mean(num_vars);
        std::vector<double> recall_mean(num_vars);
        std::vector<double> f1_mean(num_vars);
        std::vector<double> mcc_mean(num_vars);

        std::vector<double> tp_std(num_vars);
        std::vector<double> fp_std(num_vars);
        std::vector<double> fn_std(num_vars);
        std::vector<double> tn_std(num_vars);
        std::vector<double> acc_std(num_vars);
        std::vector<double> precision_std(num_vars);
        std::vector<double> recall_std(num_vars);
        std::vector<double> f1_std(num_vars);
        std::vector<double> mcc_std(num_vars);

        std::string variable = "n";

        for (int i = 0; i < num_vars; i++)
        {
            int var = 10 * i * i + 10;
            x[i] = var;
            std::pair<Stats, Stats> result = experiment(var, m, k, a, p, q, r, seed, b);
            // std::pair<Stats, Stats> result = experiment(var, m, k/std::cbrt((float) var) , a, p, q, r, seed, b, threads);
            tp_mean[i] = result.first.true_pos;
            fp_mean[i] = result.first.false_pos;
            fn_mean[i] = result.first.false_neg;
            tn_mean[i] = result.first.true_neg;
            acc_mean[i] = result.first.accuracy;
            precision_mean[i] = result.first.precision;
            recall_mean[i] = result.first.recall;
            f1_mean[i] = result.first.f1_score;
            mcc_mean[i] = result.first.mcc;

            tp_std[i] = result.second.true_pos;
            fp_std[i] = result.second.false_pos;
            fn_std[i] = result.second.false_neg;
            tn_std[i] = result.second.true_neg;
            acc_std[i] = result.second.accuracy;
            precision_std[i] = result.second.precision;
            recall_std[i] = result.second.recall;
            f1_std[i] = result.second.f1_score;
            mcc_std[i] = result.second.mcc;
        }

        Plotter plt(1200, 900, 30);
        plt.set_legend("bottom");
        // plt.set_logscale_x();
        plt.set_xlim(x[0], x.back());
        plt.set_ylim(0, 1);
        plt.set_savePath(("plots/" + variable + ".png").c_str());
        plt.set_title(("Model Performance with varying \"" + variable + "\"").c_str());
        plt.set_xlabel(("Class Strength (" + variable + ")").c_str());

        plt.createPlot(x, acc_mean, "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = acc_mean[i] + acc_std[i];
            lb[i] = acc_mean[i] - acc_std[i];
        }
        plt.fillBetween(x, ub, lb, "red", 0.1);

        plt.addPlot(x, precision_mean, "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = precision_mean[i] + precision_std[i];
            lb[i] = precision_mean[i] - precision_std[i];
        }
        plt.fillBetween(x, ub, lb, "blue", 0.1);

        plt.addPlot(x, recall_mean, "Recall", "green", Plotter::TriDF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = recall_mean[i] + recall_std[i];
            lb[i] = recall_mean[i] - recall_std[i];
        }
        plt.fillBetween(x, ub, lb, "green", 0.1);

        plt.addPlot(x, f1_mean, "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = f1_mean[i] + f1_std[i];
            lb[i] = f1_mean[i] - f1_std[i];
        }
        plt.fillBetween(x, ub, lb, "black", 0.1);

        plt.addPlot(x, mcc_mean, "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = mcc_mean[i] + mcc_std[i];
            lb[i] = mcc_mean[i] - mcc_std[i];
        }
        plt.fillBetween(x, ub, lb, "brown", 0.1);

        std::ofstream fout("plots/desc/" + variable + ".txt");
        fout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
        fout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
        fout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
        fout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
        fout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
        fout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
        fout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
        fout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
        fout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
        fout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
        fout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
        fout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
        fout << "---------------------------------------------------------------------------- " << std::endl;
        std::cout << "---------------------------------------------------------------------------- " << std::endl;

        // ************************************************ //

        variable = "p";

        for (int i = 0; i < num_vars; i++)
        {
            float var = 0.1 * i;
            x[i] = var;
            std::pair<Stats, Stats> result = experiment(n, m, k, a, var, q, r, seed, b);
            tp_mean[i] = result.first.true_pos;
            fp_mean[i] = result.first.false_pos;
            fn_mean[i] = result.first.false_neg;
            tn_mean[i] = result.first.true_neg;
            acc_mean[i] = result.first.accuracy;
            precision_mean[i] = result.first.precision;
            recall_mean[i] = result.first.recall;
            f1_mean[i] = result.first.f1_score;
            mcc_mean[i] = result.first.mcc;

            tp_std[i] = result.second.true_pos;
            fp_std[i] = result.second.false_pos;
            fn_std[i] = result.second.false_neg;
            tn_std[i] = result.second.true_neg;
            acc_std[i] = result.second.accuracy;
            precision_std[i] = result.second.precision;
            recall_std[i] = result.second.recall;
            f1_std[i] = result.second.f1_score;
            mcc_std[i] = result.second.mcc;
        }

        plt.reset(1200, 900, 30);
        plt.set_legend("bottom");
        plt.set_xlim(0, 1);
        plt.set_ylim(-1, 1);
        plt.set_savePath(("plots/" + variable + ".png").c_str());
        plt.set_title(("Model Performance with varying \"" + variable + "\"").c_str());
        plt.set_xlabel(("Prob. to mark present student present (" + variable + ")").c_str());

        plt.createPlot(x, acc_mean, "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = acc_mean[i] + acc_std[i];
            lb[i] = acc_mean[i] - acc_std[i];
        }
        plt.fillBetween(x, ub, lb, "red", 0.1);

        plt.addPlot(x, precision_mean, "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = precision_mean[i] + precision_std[i];
            lb[i] = precision_mean[i] - precision_std[i];
        }
        plt.fillBetween(x, ub, lb, "blue", 0.1);

        plt.addPlot(x, recall_mean, "Recall", "green", Plotter::TriDF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = recall_mean[i] + recall_std[i];
            lb[i] = recall_mean[i] - recall_std[i];
        }
        plt.fillBetween(x, ub, lb, "green", 0.1);

        plt.addPlot(x, f1_mean, "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = f1_mean[i] + f1_std[i];
            lb[i] = f1_mean[i] - f1_std[i];
        }
        plt.fillBetween(x, ub, lb, "black", 0.1);

        plt.addPlot(x, mcc_mean, "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = mcc_mean[i] + mcc_std[i];
            lb[i] = mcc_mean[i] - mcc_std[i];
        }
        plt.fillBetween(x, ub, lb, "brown", 0.1);

        fout = std::ofstream("plots/desc/" + variable + ".txt");
        fout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
        fout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
        fout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
        fout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
        fout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
        fout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
        fout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
        fout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
        fout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
        fout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
        fout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
        fout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
        fout << "---------------------------------------------------------------------------- " << std::endl;
        std::cout << "---------------------------------------------------------------------------- " << std::endl;

        // ************************************************ //

        variable = "q";

        for (int i = 0; i < num_vars; i++)
        {
            float var = 0.1 * i;
            x[i] = var;
            std::pair<Stats, Stats> result = experiment(n, m, k, a, p, var, r, seed, b);
            tp_mean[i] = result.first.true_pos;
            fp_mean[i] = result.first.false_pos;
            fn_mean[i] = result.first.false_neg;
            tn_mean[i] = result.first.true_neg;
            acc_mean[i] = result.first.accuracy;
            precision_mean[i] = result.first.precision;
            recall_mean[i] = result.first.recall;
            f1_mean[i] = result.first.f1_score;
            mcc_mean[i] = result.first.mcc;

            tp_std[i] = result.second.true_pos;
            fp_std[i] = result.second.false_pos;
            fn_std[i] = result.second.false_neg;
            tn_std[i] = result.second.true_neg;
            acc_std[i] = result.second.accuracy;
            precision_std[i] = result.second.precision;
            recall_std[i] = result.second.recall;
            f1_std[i] = result.second.f1_score;
            mcc_std[i] = result.second.mcc;
        }

        plt.reset(1200, 900, 30);
        plt.set_legend("bottom left");
        plt.set_xlim(0, 1);
        plt.set_ylim(-1, 1);
        plt.set_savePath(("plots/" + variable + ".png").c_str());
        plt.set_title(("Model Performance with varying \"" + variable + "\"").c_str());
        plt.set_xlabel(("Prob. to mark absent student present (" + variable + ")").c_str());

        plt.createPlot(x, acc_mean, "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = acc_mean[i] + acc_std[i];
            lb[i] = acc_mean[i] - acc_std[i];
        }
        plt.fillBetween(x, ub, lb, "red", 0.1);

        plt.addPlot(x, precision_mean, "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = precision_mean[i] + precision_std[i];
            lb[i] = precision_mean[i] - precision_std[i];
        }
        plt.fillBetween(x, ub, lb, "blue", 0.1);

        plt.addPlot(x, recall_mean, "Recall", "green", Plotter::TriDF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = recall_mean[i] + recall_std[i];
            lb[i] = recall_mean[i] - recall_std[i];
        }
        plt.fillBetween(x, ub, lb, "green", 0.1);

        plt.addPlot(x, f1_mean, "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = f1_mean[i] + f1_std[i];
            lb[i] = f1_mean[i] - f1_std[i];
        }
        plt.fillBetween(x, ub, lb, "black", 0.1);

        plt.addPlot(x, mcc_mean, "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = mcc_mean[i] + mcc_std[i];
            lb[i] = mcc_mean[i] - mcc_std[i];
        }
        plt.fillBetween(x, ub, lb, "brown", 0.1);

        fout = std::ofstream("plots/desc/" + variable + ".txt");
        fout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
        fout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
        fout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
        fout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
        fout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
        fout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
        fout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
        fout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
        fout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
        fout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
        fout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
        fout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
        fout << "---------------------------------------------------------------------------- " << std::endl;
        std::cout << "---------------------------------------------------------------------------- " << std::endl;

        // ************************************************ //

        variable = "r";

        for (int i = 0; i < num_vars; i++)
        {
            float var = 0.05 * (i + 2);
            x[i] = var;
            std::pair<Stats, Stats> result = experiment(n, m, k, a, p, q, var, seed, b);
            tp_mean[i] = result.first.true_pos;
            fp_mean[i] = result.first.false_pos;
            fn_mean[i] = result.first.false_neg;
            tn_mean[i] = result.first.true_neg;
            acc_mean[i] = result.first.accuracy;
            precision_mean[i] = result.first.precision;
            recall_mean[i] = result.first.recall;
            f1_mean[i] = result.first.f1_score;
            mcc_mean[i] = result.first.mcc;

            tp_std[i] = result.second.true_pos;
            fp_std[i] = result.second.false_pos;
            fn_std[i] = result.second.false_neg;
            tn_std[i] = result.second.true_neg;
            acc_std[i] = result.second.accuracy;
            precision_std[i] = result.second.precision;
            recall_std[i] = result.second.recall;
            f1_std[i] = result.second.f1_score;
            mcc_std[i] = result.second.mcc;
        }

        plt.reset(1200, 900, 30);
        plt.set_legend("bottom");
        // plt.set_logscale_x();
        plt.set_xlim(x[0], x.back());
        plt.set_ylim(0, 1);
        plt.set_savePath(("plots/" + variable + ".png").c_str());
        plt.set_title(("Model Performance with varying \"" + variable + "\"").c_str());
        plt.set_xlabel(("Porb. that a student is present (" + variable + ")").c_str());

        plt.createPlot(x, acc_mean, "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = acc_mean[i] + acc_std[i];
            lb[i] = acc_mean[i] - acc_std[i];
        }
        plt.fillBetween(x, ub, lb, "red", 0.1);

        plt.addPlot(x, precision_mean, "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = precision_mean[i] + precision_std[i];
            lb[i] = precision_mean[i] - precision_std[i];
        }
        plt.fillBetween(x, ub, lb, "blue", 0.1);

        plt.addPlot(x, recall_mean, "Recall", "green", Plotter::TriDF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = recall_mean[i] + recall_std[i];
            lb[i] = recall_mean[i] - recall_std[i];
        }
        plt.fillBetween(x, ub, lb, "green", 0.1);

        plt.addPlot(x, f1_mean, "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = f1_mean[i] + f1_std[i];
            lb[i] = f1_mean[i] - f1_std[i];
        }
        plt.fillBetween(x, ub, lb, "black", 0.1);

        plt.addPlot(x, mcc_mean, "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = mcc_mean[i] + mcc_std[i];
            lb[i] = mcc_mean[i] - mcc_std[i];
        }
        plt.fillBetween(x, ub, lb, "brown", 0.1);

        fout = std::ofstream("plots/desc/" + variable + ".txt");
        fout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
        fout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
        fout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
        fout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
        fout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
        fout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
        fout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
        fout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
        fout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
        fout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
        fout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
        fout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
        fout << "---------------------------------------------------------------------------- " << std::endl;
        std::cout << "---------------------------------------------------------------------------- " << std::endl;

        // ************************************************ //

        variable = "m";

        for (int i = 0; i < num_vars; i++)
        {
            float var = n / (i + 2);
            if (i > 0 && var >= x[i - 1] * n - 1)
                var = x[i - 1] * n - 2;
            var /= n;
            x[i] = var;
            std::pair<Stats, Stats> result = experiment(n, var, k, a, p, q, r, seed, b);
            tp_mean[i] = result.first.true_pos;
            fp_mean[i] = result.first.false_pos;
            fn_mean[i] = result.first.false_neg;
            tn_mean[i] = result.first.true_neg;
            acc_mean[i] = result.first.accuracy;
            precision_mean[i] = result.first.precision;
            recall_mean[i] = result.first.recall;
            f1_mean[i] = result.first.f1_score;
            mcc_mean[i] = result.first.mcc;

            tp_std[i] = result.second.true_pos;
            fp_std[i] = result.second.false_pos;
            fn_std[i] = result.second.false_neg;
            tn_std[i] = result.second.true_neg;
            acc_std[i] = result.second.accuracy;
            precision_std[i] = result.second.precision;
            recall_std[i] = result.second.recall;
            f1_std[i] = result.second.f1_score;
            mcc_std[i] = result.second.mcc;
        }

        plt.reset(1200, 900, 30);
        plt.set_legend("bottom");
        // plt.set_logscale_x();
        plt.set_xlim(x.back(), x[0]);
        plt.set_ylim(0, 1);
        plt.set_savePath(("plots/" + variable + ".png").c_str());
        plt.set_title(("Model Performance with varying \"" + variable + "\"").c_str());
        plt.set_xlabel(("Fraction of Students to roll call (" + variable + ")").c_str());

        plt.createPlot(x, acc_mean, "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = acc_mean[i] + acc_std[i];
            lb[i] = acc_mean[i] - acc_std[i];
        }
        plt.fillBetween(x, ub, lb, "red", 0.1);

        plt.addPlot(x, precision_mean, "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = precision_mean[i] + precision_std[i];
            lb[i] = precision_mean[i] - precision_std[i];
        }
        plt.fillBetween(x, ub, lb, "blue", 0.1);

        plt.addPlot(x, recall_mean, "Recall", "green", Plotter::TriDF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = recall_mean[i] + recall_std[i];
            lb[i] = recall_mean[i] - recall_std[i];
        }
        plt.fillBetween(x, ub, lb, "green", 0.1);

        plt.addPlot(x, f1_mean, "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = f1_mean[i] + f1_std[i];
            lb[i] = f1_mean[i] - f1_std[i];
        }
        plt.fillBetween(x, ub, lb, "black", 0.1);

        plt.addPlot(x, mcc_mean, "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = mcc_mean[i] + mcc_std[i];
            lb[i] = mcc_mean[i] - mcc_std[i];
        }
        plt.fillBetween(x, ub, lb, "brown", 0.1);

        fout = std::ofstream("plots/desc/" + variable + ".txt");
        fout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
        fout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
        fout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
        fout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
        fout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
        fout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
        fout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
        fout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
        fout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
        fout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
        fout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
        fout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
        fout << "---------------------------------------------------------------------------- " << std::endl;
        std::cout << "---------------------------------------------------------------------------- " << std::endl;

        // ************************************************ //

        variable = "k";

        for (int i = 0; i < num_vars; i++)
        {
            float var = 2 * a * n / (i + 2);
            if (i > 0 && var >= x[i - 1] * n - 1)
                var = x[i - 1] * n - 2;
            var /= n;
            x[i] = var;
            std::pair<Stats, Stats> result = experiment(n, m, var, a, p, q, r, seed, b);
            tp_mean[i] = result.first.true_pos;
            fp_mean[i] = result.first.false_pos;
            fn_mean[i] = result.first.false_neg;
            tn_mean[i] = result.first.true_neg;
            acc_mean[i] = result.first.accuracy;
            precision_mean[i] = result.first.precision;
            recall_mean[i] = result.first.recall;
            f1_mean[i] = result.first.f1_score;
            mcc_mean[i] = result.first.mcc;

            tp_std[i] = result.second.true_pos;
            fp_std[i] = result.second.false_pos;
            fn_std[i] = result.second.false_neg;
            tn_std[i] = result.second.true_neg;
            acc_std[i] = result.second.accuracy;
            precision_std[i] = result.second.precision;
            recall_std[i] = result.second.recall;
            f1_std[i] = result.second.f1_score;
            mcc_std[i] = result.second.mcc;
        }

        plt.reset(1200, 900, 30);
        plt.set_legend("bottom left");
        // plt.set_logscale_x();
        plt.set_xlim(x.back(), x[0]);
        plt.set_ylim(0, 1);
        plt.set_savePath(("plots/" + variable + ".png").c_str());
        plt.set_title(("Model Performance with varying \"" + variable + "\"").c_str());
        plt.set_xlabel(("Min. Number of Votes (frac. of n) required to be present (" + variable + ")").c_str());

        plt.createPlot(x, acc_mean, "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = acc_mean[i] + acc_std[i];
            lb[i] = acc_mean[i] - acc_std[i];
        }
        plt.fillBetween(x, ub, lb, "red", 0.1);

        plt.addPlot(x, precision_mean, "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = precision_mean[i] + precision_std[i];
            lb[i] = precision_mean[i] - precision_std[i];
        }
        plt.fillBetween(x, ub, lb, "blue", 0.1);

        plt.addPlot(x, recall_mean, "Recall", "green", Plotter::TriDF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = recall_mean[i] + recall_std[i];
            lb[i] = recall_mean[i] - recall_std[i];
        }
        plt.fillBetween(x, ub, lb, "green", 0.1);

        plt.addPlot(x, f1_mean, "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = f1_mean[i] + f1_std[i];
            lb[i] = f1_mean[i] - f1_std[i];
        }
        plt.fillBetween(x, ub, lb, "black", 0.1);

        plt.addPlot(x, mcc_mean, "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = mcc_mean[i] + mcc_std[i];
            lb[i] = mcc_mean[i] - mcc_std[i];
        }
        plt.fillBetween(x, ub, lb, "brown", 0.1);

        fout = std::ofstream("plots/desc/" + variable + ".txt");
        fout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
        fout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
        fout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
        fout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
        fout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
        fout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
        fout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
        fout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
        fout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
        fout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
        fout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
        fout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
        fout << "---------------------------------------------------------------------------- " << std::endl;
        std::cout << "---------------------------------------------------------------------------- " << std::endl;

        // ************************************************ //

        variable = "b";

        for (int i = 0; i < num_vars; i++)
        {
            float var = 2 * a * n / (i + 2);
            if (i > 0 && var >= x[i - 1] * n - 1)
                var = x[i - 1] * n - 2;
            var /= n;
            x[i] = var;
            std::pair<Stats, Stats> result = experiment(n, m, k, a, p, q, r, seed, var);
            tp_mean[i] = result.first.true_pos;
            fp_mean[i] = result.first.false_pos;
            fn_mean[i] = result.first.false_neg;
            tn_mean[i] = result.first.true_neg;
            acc_mean[i] = result.first.accuracy;
            precision_mean[i] = result.first.precision;
            recall_mean[i] = result.first.recall;
            f1_mean[i] = result.first.f1_score;
            mcc_mean[i] = result.first.mcc;

            tp_std[i] = result.second.true_pos;
            fp_std[i] = result.second.false_pos;
            fn_std[i] = result.second.false_neg;
            tn_std[i] = result.second.true_neg;
            acc_std[i] = result.second.accuracy;
            precision_std[i] = result.second.precision;
            recall_std[i] = result.second.recall;
            f1_std[i] = result.second.f1_score;
            mcc_std[i] = result.second.mcc;
        }

        plt.reset(1200, 900, 30);
        plt.set_legend("bottom");
        // plt.set_logscale_x();
        plt.set_xlim(x.back(), x[0]);
        plt.set_ylim(0, 1);
        plt.set_savePath(("plots/" + variable + ".png").c_str());
        plt.set_title(("Model Performance with varying \"" + variable + "\"").c_str());
        plt.set_xlabel(("Penalty (frac. of n) for voting an absent student (" + variable + ")").c_str());

        plt.createPlot(x, acc_mean, "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = acc_mean[i] + acc_std[i];
            lb[i] = acc_mean[i] - acc_std[i];
        }
        plt.fillBetween(x, ub, lb, "red", 0.1);

        plt.addPlot(x, precision_mean, "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = precision_mean[i] + precision_std[i];
            lb[i] = precision_mean[i] - precision_std[i];
        }
        plt.fillBetween(x, ub, lb, "blue", 0.1);

        plt.addPlot(x, recall_mean, "Recall", "green", Plotter::TriDF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = recall_mean[i] + recall_std[i];
            lb[i] = recall_mean[i] - recall_std[i];
        }
        plt.fillBetween(x, ub, lb, "green", 0.1);

        plt.addPlot(x, f1_mean, "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = f1_mean[i] + f1_std[i];
            lb[i] = f1_mean[i] - f1_std[i];
        }
        plt.fillBetween(x, ub, lb, "black", 0.1);

        plt.addPlot(x, mcc_mean, "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = mcc_mean[i] + mcc_std[i];
            lb[i] = mcc_mean[i] - mcc_std[i];
        }
        plt.fillBetween(x, ub, lb, "brown", 0.1);

        fout = std::ofstream("plots/desc/" + variable + ".txt");
        fout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
        fout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
        fout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
        fout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
        fout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
        fout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
        fout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
        fout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
        fout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
        fout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
        fout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
        fout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
        fout << "---------------------------------------------------------------------------- " << std::endl;
        std::cout << "---------------------------------------------------------------------------- " << std::endl;

        // ************************************************ //

        variable = "a";

        for (int i = 0; i < num_vars; i++)
        {
            float var = n / (i + 2) + k;
            if (i > 0 && var >= x[i - 1] * n - 1)
                var = x[i - 1] * n - 2;
            var /= n;
            x[i] = var;
            std::pair<Stats, Stats> result = experiment(n, m, k, var, p, q, r, seed, b);
            tp_mean[i] = result.first.true_pos;
            fp_mean[i] = result.first.false_pos;
            fn_mean[i] = result.first.false_neg;
            tn_mean[i] = result.first.true_neg;
            acc_mean[i] = result.first.accuracy;
            precision_mean[i] = result.first.precision;
            recall_mean[i] = result.first.recall;
            f1_mean[i] = result.first.f1_score;
            mcc_mean[i] = result.first.mcc;

            tp_std[i] = result.second.true_pos;
            fp_std[i] = result.second.false_pos;
            fn_std[i] = result.second.false_neg;
            tn_std[i] = result.second.true_neg;
            acc_std[i] = result.second.accuracy;
            precision_std[i] = result.second.precision;
            recall_std[i] = result.second.recall;
            f1_std[i] = result.second.f1_score;
            mcc_std[i] = result.second.mcc;
        }

        plt.reset(1200, 900, 30);
        plt.set_legend("bottom");
        // plt.set_logscale_x();
        plt.set_xlim(x.back(), x[0]);
        plt.set_ylim(0, 1);
        plt.set_savePath(("plots/" + variable + ".png").c_str());
        plt.set_title(("Model Performance with varying \"" + variable + "\"").c_str());
        plt.set_xlabel(("Max. Fraction of Students to ask for vote (" + variable + ")").c_str());

        plt.createPlot(x, acc_mean, "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = acc_mean[i] + acc_std[i];
            lb[i] = acc_mean[i] - acc_std[i];
        }
        plt.fillBetween(x, ub, lb, "red", 0.1);

        plt.addPlot(x, precision_mean, "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = precision_mean[i] + precision_std[i];
            lb[i] = precision_mean[i] - precision_std[i];
        }
        plt.fillBetween(x, ub, lb, "blue", 0.1);

        plt.addPlot(x, recall_mean, "Recall", "green", Plotter::TriDF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = recall_mean[i] + recall_std[i];
            lb[i] = recall_mean[i] - recall_std[i];
        }
        plt.fillBetween(x, ub, lb, "green", 0.1);

        plt.addPlot(x, f1_mean, "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = f1_mean[i] + f1_std[i];
            lb[i] = f1_mean[i] - f1_std[i];
        }
        plt.fillBetween(x, ub, lb, "black", 0.1);

        plt.addPlot(x, mcc_mean, "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
        for (int i = 0; i < num_vars; i++)
        {
            ub[i] = mcc_mean[i] + mcc_std[i];
            lb[i] = mcc_mean[i] - mcc_std[i];
        }
        plt.fillBetween(x, ub, lb, "brown", 0.1);

        fout = std::ofstream("plots/desc/" + variable + ".txt");
        fout << "########################## EXPERIMENT PARAMETERS ########################### " << std::endl;
        fout << "Number of Threads -------------------------------------------- threads: " << threads << std::endl;
        fout << "Seed (Random Number Generator) ---------------------------------- seed: " << seed << std::endl;
        fout << "Number of Simmulations ------------------------------------------- sim: " << sim << std::endl;
        fout << "Number of Students in the Class ------------------------------------ n: " << n << std::endl;
        fout << "Fraction of Students to pick for rollcall -------------------------- m: " << m << std::endl;
        fout << "Minimum Number of Votes required to be present (as Fraction of n) -- k: " << k << std::endl;
        fout << "Maximum Fraction of students that each student can ask for vote ---- a: " << a << std::endl;
        fout << "Penalty for voting a absent student (as Fraction of n) ------------- b: " << b << std::endl;
        fout << "Probability that a student is marked present if he/she is present -- p: " << p << std::endl;
        fout << "Probability that a student is marked present if he/she is absent --- q: " << q << std::endl;
        fout << "Probability that a student is present in the class ----------------- r: " << r << std::endl;
        fout << "---------------------------------------------------------------------------- " << std::endl;
        std::cout << "---------------------------------------------------------------------------- " << std::endl;
    }
    else
    {
        std::vector<std::vector<double>> tp_mean(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> fp_mean(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> fn_mean(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> tn_mean(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> acc_mean(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> precision_mean(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> recall_mean(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> f1_mean(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> mcc_mean(5, std::vector<double>(num_vars, 0));

        std::vector<std::vector<double>> tp_std(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> fp_std(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> fn_std(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> tn_std(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> acc_std(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> precision_std(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> recall_std(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> f1_std(5, std::vector<double>(num_vars, 0));
        std::vector<std::vector<double>> mcc_std(5, std::vector<double>(num_vars, 0));

        std::vector<std::vector<int>> best_a(5, std::vector<int>(num_vars));
        std::vector<std::vector<int>> best_k(5, std::vector<int>(num_vars));
        std::vector<std::vector<int>> best_m(5, std::vector<int>(num_vars));
        std::vector<std::vector<int>> best_b(5, std::vector<int>(num_vars));

        std::string variable = "n";

        int cnt = 0;

        for (int i = 0; i < num_vars; i++)
        {
            int n = 10 * i * i + 10;
            x[i] = n;

            std::vector<std::vector<int>> controls;

            double log_n = std::log(n);
            float a = 20.0 / (log_n * log_n * log_n);
            if (a > 0.9)
                a = 0.9;
            // float a = 2.0 / std::log(n);
            for (float m = 0.02; m < 0.11; m += 0.02)
                for (float k = 0.1 * a; k < a * 0.9; k += 0.09 * a)
                    for (float b = 0.1 * a; b < a * 0.9; b += 0.09 * a)
                    {
                        controls.push_back({static_cast<int>(std::round(a * n)),
                                            static_cast<int>(std::round(m * n)),
                                            static_cast<int>(std::round(k * n)),
                                            static_cast<int>(std::round(b * n))});
                    }

            std::vector<std::vector<std::pair<Stats, Stats>>> best_result(threads, std::vector<std::pair<Stats, Stats>>(5, {Stats(), Stats()}));
            std::vector<std::vector<std::vector<int>>> best_control(threads, std::vector<std::vector<int>>(5));

#pragma omp parallel for num_threads(threads)
            for (std::vector<int> v : controls)
            {
                cnt++;
                int t = omp_get_thread_num();
                // std::cout << "Thread " << t << " : " << i << ", " << cnt << std::endl;

                a = ((float)v[0]) / ((float)n);
                m = ((float)v[1]) / ((float)n);
                k = ((float)v[2]) / ((float)n);
                b = ((float)v[3]) / ((float)n);
                std::pair<Stats, Stats> result = experiment(n, m, k, a, p, q, r, seed, b);
                if (result.first.accuracy > best_result[t][0].first.accuracy)
                {
                    best_result[t][0] = result;
                    best_control[t][0] = v;
                }
                if (result.first.precision > best_result[t][1].first.precision)
                {
                    best_result[t][1] = result;
                    best_control[t][1] = v;
                }
                if (result.first.recall > best_result[t][2].first.recall)
                {
                    best_result[t][2] = result;
                    best_control[t][2] = v;
                }
                if (result.first.f1_score > best_result[t][3].first.f1_score)
                {
                    best_result[t][3] = result;
                    best_control[t][3] = v;
                }
                if (result.first.mcc > best_result[t][4].first.mcc)
                {
                    best_result[t][4] = result;
                    best_control[t][4] = v;
                }
            }

            for (int t = 0; t < threads; t++)
            {
                if (best_result[t][0].first.accuracy > acc_mean[0][i])
                {
                    acc_mean[0][i] = best_result[t][0].first.accuracy;
                    precision_mean[0][i] = best_result[t][0].first.precision;
                    recall_mean[0][i] = best_result[t][0].first.recall;
                    f1_mean[0][i] = best_result[t][0].first.f1_score;
                    mcc_mean[0][i] = best_result[t][0].first.mcc;

                    acc_std[0][i] = best_result[t][0].second.accuracy;
                    precision_std[0][i] = best_result[t][0].second.precision;
                    recall_std[0][i] = best_result[t][0].second.recall;
                    f1_std[0][i] = best_result[t][0].second.f1_score;
                    mcc_std[0][i] = best_result[t][0].second.mcc;

                    best_a[0][i] = best_control[t][0][0];
                    best_m[0][i] = best_control[t][0][1];
                    best_k[0][i] = best_control[t][0][2];
                    best_b[0][i] = best_control[t][0][3];
                }
                if (best_result[t][1].first.precision > precision_mean[1][i])
                {
                    acc_mean[1][i] = best_result[t][1].first.accuracy;
                    precision_mean[1][i] = best_result[t][1].first.precision;
                    recall_mean[1][i] = best_result[t][1].first.recall;
                    f1_mean[1][i] = best_result[t][1].first.f1_score;
                    mcc_mean[1][i] = best_result[t][1].first.mcc;

                    acc_std[1][i] = best_result[t][1].second.accuracy;
                    precision_std[1][i] = best_result[t][1].second.precision;
                    recall_std[1][i] = best_result[t][1].second.recall;
                    f1_std[1][i] = best_result[t][1].second.f1_score;
                    mcc_std[1][i] = best_result[t][1].second.mcc;

                    best_a[1][i] = best_control[t][1][0];
                    best_m[1][i] = best_control[t][1][1];
                    best_k[1][i] = best_control[t][1][2];
                    best_b[1][i] = best_control[t][1][3];
                }
                if (best_result[t][2].first.recall > recall_mean[2][i])
                {
                    acc_mean[2][i] = best_result[t][2].first.accuracy;
                    precision_mean[2][i] = best_result[t][2].first.precision;
                    recall_mean[2][i] = best_result[t][2].first.recall;
                    f1_mean[2][i] = best_result[t][2].first.f1_score;
                    mcc_mean[2][i] = best_result[t][2].first.mcc;

                    acc_std[2][i] = best_result[t][2].second.accuracy;
                    precision_std[2][i] = best_result[t][2].second.precision;
                    recall_std[2][i] = best_result[t][2].second.recall;
                    f1_std[2][i] = best_result[t][2].second.f1_score;
                    mcc_std[2][i] = best_result[t][2].second.mcc;

                    best_a[2][i] = best_control[t][2][0];
                    best_m[2][i] = best_control[t][2][1];
                    best_k[2][i] = best_control[t][2][2];
                    best_b[2][i] = best_control[t][2][3];
                }
                if (best_result[t][3].first.f1_score > f1_mean[3][i])
                {
                    acc_mean[3][i] = best_result[t][3].first.accuracy;
                    precision_mean[3][i] = best_result[t][3].first.precision;
                    recall_mean[3][i] = best_result[t][3].first.recall;
                    f1_mean[3][i] = best_result[t][3].first.f1_score;
                    mcc_mean[3][i] = best_result[t][3].first.mcc;

                    acc_std[3][i] = best_result[t][3].second.accuracy;
                    precision_std[3][i] = best_result[t][3].second.precision;
                    recall_std[3][i] = best_result[t][3].second.recall;
                    f1_std[3][i] = best_result[t][3].second.f1_score;
                    mcc_std[3][i] = best_result[t][3].second.mcc;

                    best_a[3][i] = best_control[t][3][0];
                    best_m[3][i] = best_control[t][3][1];
                    best_k[3][i] = best_control[t][3][2];
                    best_b[3][i] = best_control[t][3][3];
                }
                if (best_result[t][4].first.mcc > mcc_mean[4][i])
                {
                    acc_mean[4][i] = best_result[t][4].first.accuracy;
                    precision_mean[4][i] = best_result[t][4].first.precision;
                    recall_mean[4][i] = best_result[t][4].first.recall;
                    f1_mean[4][i] = best_result[t][4].first.f1_score;
                    mcc_mean[4][i] = best_result[t][4].first.mcc;

                    acc_std[4][i] = best_result[t][4].second.accuracy;
                    precision_std[4][i] = best_result[t][4].second.precision;
                    recall_std[4][i] = best_result[t][4].second.recall;
                    f1_std[4][i] = best_result[t][4].second.f1_score;
                    mcc_std[4][i] = best_result[t][4].second.mcc;

                    best_a[4][i] = best_control[t][4][0];
                    best_m[4][i] = best_control[t][4][1];
                    best_k[4][i] = best_control[t][4][2];
                    best_b[4][i] = best_control[t][4][3];
                }
            }
        }

        Plotter plt(1200, 900, 30);
        plt.set_legend("bottom right");
        plt.set_logscale_x();
        plt.set_xlim(x[0], x.back());
        plt.set_ylim(0, 1);
        plt.set_xlabel(("Class Strength (" + variable + ")").c_str());

        // ************************************************ //
        std::ofstream fout;
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << p;
        std::string p_str = stream.str();
        stream.str("");
        stream << std::fixed << std::setprecision(2) << q;
        std::string q_str = stream.str();
        stream.str("");
        stream << std::fixed << std::setprecision(2) << r;
        std::string r_str = stream.str();
        for (int j = 0; j < 5; j++)
        {
            plt.set_legend("bottom right");
            plt.set_logscale_x();
            plt.set_ylim(0, 1);
            if (j == 0)
            {
                plt.set_title("Model Performance when maximizing \"Accuracy\"");
                plt.set_savePath(("plots/acc_metric_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
                fout = std::ofstream("plots/desc/accuracy_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".txt");
            }
            else if (j == 1)
            {
                plt.set_title("Model Performance when maximizing \"Precision\"");
                plt.set_savePath(("plots/prec_metric_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
                fout = std::ofstream("plots/desc/precision_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".txt");
            }
            else if (j == 2)
            {
                plt.set_title("Model Performance when maximizing \"Recall\"");
                plt.set_savePath(("plots/rec_metric_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
                fout = std::ofstream("plots/desc/recall_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".txt");
            }
            else if (j == 3)
            {
                plt.set_title("Model Performance when maximizing \"F1-Score\"");
                plt.set_savePath(("plots/f1_metric_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
                fout = std::ofstream("plots/desc/f1-score_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".txt");
            }
            else
            {
                plt.set_title("Model Performance when maximizing \"MCC\"");
                plt.set_savePath(("plots/mcc_metric_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
                fout = std::ofstream("plots/desc/mcc_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".txt");
            }

            plt.createPlot(x, acc_mean[j], "Accuracy", "red", Plotter::CircleF, 1.0, 3.0);
            for (int i = 0; i < num_vars; i++)
            {
                ub[i] = acc_mean[j][i] + acc_std[j][i];
                lb[i] = acc_mean[j][i] - acc_std[j][i];
            }
            plt.fillBetween(x, ub, lb, "red", 0.1);

            plt.addPlot(x, precision_mean[j], "Precision", "blue", Plotter::BoxF, 1.0, 3.0);
            for (int i = 0; i < num_vars; i++)
            {
                ub[i] = precision_mean[j][i] + precision_std[j][i];
                lb[i] = precision_mean[j][i] - precision_std[j][i];
            }
            plt.fillBetween(x, ub, lb, "blue", 0.1);

            plt.addPlot(x, recall_mean[j], "Recall", "green", Plotter::TriDF, 1.0, 3.0);
            for (int i = 0; i < num_vars; i++)
            {
                ub[i] = recall_mean[j][i] + recall_std[j][i];
                lb[i] = recall_mean[j][i] - recall_std[j][i];
            }
            plt.fillBetween(x, ub, lb, "green", 0.1);

            plt.addPlot(x, f1_mean[j], "F1 Score", "black", Plotter::TriUF, 1.0, 3.0);
            for (int i = 0; i < num_vars; i++)
            {
                ub[i] = f1_mean[j][i] + f1_std[j][i];
                lb[i] = f1_mean[j][i] - f1_std[j][i];
            }
            plt.fillBetween(x, ub, lb, "black", 0.1);

            plt.addPlot(x, mcc_mean[j], "MCC", "brown", Plotter::DiaF, 1.0, 3.0);
            for (int i = 0; i < num_vars; i++)
            {
                ub[i] = mcc_mean[j][i] + mcc_std[j][i];
                lb[i] = mcc_mean[j][i] - mcc_std[j][i];
            }
            plt.fillBetween(x, ub, lb, "brown", 0.1);

            plt.plot();

            // ************************************************ //

            plt.unset_logscale_x();
            plt.set_legend("top left");
            int max_a = best_a[j][0];
            for (int aBest : best_a[j])
                max_a = std::max(max_a, aBest);
            plt.set_ylim(0, max_a + 10);
            if (j == 0)
            {
                plt.set_title("Best Control Variables when maximizing \"Accuracy\"");
                plt.set_savePath(("plots/acc_control_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
            }
            else if (j == 1)
            {
                plt.set_title("Best Control Variables when maximizing \"Precision\"");
                plt.set_savePath(("plots/prec_control_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
            }
            else if (j == 2)
            {
                plt.set_title("Best Control Variables when maximizing \"Recall\"");
                plt.set_savePath(("plots/rec_control_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
            }
            else if (j == 3)
            {
                plt.set_title("Best Control Variables when maximizing \"F1-Score\"");
                plt.set_savePath(("plots/f1_control_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
            }
            else
            {
                plt.set_title("Best Control Variables when maximizing \"MCC\"");
                plt.set_savePath(("plots/mcc_control_p-" + p_str + "_q-" + q_str + "_r-" + r_str + ".png").c_str());
            }

            plt.createPlot(x, best_a[j], "a", "red", Plotter::CircleF, 1.0, 3.0);
            plt.addPlot(x, best_m[j], "m", "blue", Plotter::BoxF, 1.0, 3.0);
            plt.addPlot(x, best_b[j], "b", "green", Plotter::TriDF, 1.0, 3.0);
            plt.addPlot(x, best_k[j], "k", "black", Plotter::TriUF, 1.0, 3.0);
            plt.plot();

            fout << "n,a,m,k,b" << std::endl;
            for (int i = 0; i < x.size(); i++)
                fout << x[i] << "," << best_a[j][i] << "," << best_m[j][i] << "," << best_k[j][i] << "," << best_b[j][i] << std::endl;
        }

        fitLinear(x, best_a, "a");
        fitLinear(x, best_m, "m");
        fitLinear(x, best_k, "k");
        fitLinear(x, best_b, "b");
    }

    return 0;
}
