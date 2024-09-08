#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

struct Stats
{
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    double mcc;

    inline Stats()
    {
        this->accuracy = 0;
        this->precision = 0;
        this->recall = 0;
        this->f1_score = 0;
        this->mcc = 0;
    }

    inline Stats(double accuracy, double precision, double recall, double f1_score, double mcc)
    {
        this->accuracy = accuracy;
        this->precision = precision;
        this->recall = recall;
        this->f1_score = f1_score;
        this->mcc = mcc;
    }

    inline Stats calc_stats(std::vector<bool> &gold, std::vector<bool> &pred)
    {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < gold.size(); i++)
        {
            if (gold[i] && pred[i])
                tp++;
            else if (!gold[i] && !pred[i])
                tn++;
            else if (gold[i] && !pred[i])
                fn++;
            else if (!gold[i] && pred[i])
                fp++;
        }
        this->accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
        this->precision = (double)tp / (tp + fp);
        this->recall = (double)tp / (tp + fn);
        this->f1_score = (double)(2 * tp) / (2 * tp + fn + fp);
        this->mcc = (double)(tp * tn - fp * fn) / std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        return *this;
    }

    inline Stats operator++()
    {
        this->accuracy++;
        this->precision++;
        this->recall++;
        this->f1_score++;
        this->mcc++;
        return *this;
    }

    inline Stats operator--()
    {
        this->accuracy++;
        this->precision++;
        this->recall++;
        this->f1_score++;
        this->mcc++;
        return *this;
    }

    inline Stats operator-() const
    {
        return Stats(
            -this->accuracy,
            -this->precision,
            -this->recall,
            -this->f1_score,
            -this->mcc);
    }

    inline Stats operator+(const double increment) const
    {
        return Stats(
            this->accuracy + increment,
            this->precision + increment,
            this->recall + increment,
            this->f1_score + increment,
            this->mcc + increment);
    }

    inline Stats operator+(const Stats &other) const
    {
        return Stats(
            this->accuracy + other.accuracy,
            this->precision + other.precision,
            this->recall + other.recall,
            this->f1_score + other.f1_score,
            this->mcc + other.mcc);
    }

    inline Stats operator-(const double decrement) const
    {
        return Stats(
            this->accuracy - decrement,
            this->precision - decrement,
            this->recall - decrement,
            this->f1_score - decrement,
            this->mcc - decrement);
    }

    inline Stats operator-(const Stats &other) const
    {
        return Stats(
            this->accuracy - other.accuracy,
            this->precision - other.precision,
            this->recall - other.recall,
            this->f1_score - other.f1_score,
            this->mcc - other.mcc);
    }

    inline Stats operator*(const double multiplier) const
    {
        return Stats(
            this->accuracy * multiplier,
            this->precision * multiplier,
            this->recall * multiplier,
            this->f1_score * multiplier,
            this->mcc * multiplier);
    }

    inline Stats operator*(const Stats &other) const
    {
        return Stats(
            this->accuracy * other.accuracy,
            this->precision * other.precision,
            this->recall * other.recall,
            this->f1_score * other.f1_score,
            this->mcc * other.mcc);
    }

    inline Stats operator/(const double divisor) const
    {
        return Stats(
            this->accuracy / divisor,
            this->precision / divisor,
            this->recall / divisor,
            this->f1_score / divisor,
            this->mcc / divisor);
    }

    inline Stats operator/(const Stats &other) const
    {
        return Stats(
            this->accuracy / other.accuracy,
            this->precision / other.precision,
            this->recall / other.recall,
            this->f1_score / other.f1_score,
            this->mcc / other.mcc);
    }

    inline Stats operator+=(const double increment)
    {
        this->accuracy += increment;
        this->precision += increment;
        this->recall += increment;
        this->f1_score += increment;
        this->mcc += increment;
        return *this;
    }

    inline Stats operator+=(const Stats &other)
    {
        this->accuracy += other.accuracy;
        this->precision += other.precision;
        this->recall += other.recall;
        this->f1_score += other.f1_score;
        this->mcc += other.mcc;
        return *this;
    }

    inline Stats operator-=(const double decrement)
    {
        this->accuracy -= decrement;
        this->precision -= decrement;
        this->recall -= decrement;
        this->f1_score -= decrement;
        this->mcc -= decrement;
        return *this;
    }

    inline Stats operator-=(const Stats &other)
    {
        this->accuracy -= other.accuracy;
        this->precision -= other.precision;
        this->recall -= other.recall;
        this->f1_score -= other.f1_score;
        this->mcc -= other.mcc;
        return *this;
    }

    inline Stats operator*=(const double multiplier)
    {
        this->accuracy *= multiplier;
        this->precision *= multiplier;
        this->recall *= multiplier;
        this->f1_score *= multiplier;
        this->mcc *= multiplier;
        return *this;
    }

    inline Stats operator*=(const Stats &other)
    {
        this->accuracy *= other.accuracy;
        this->precision *= other.precision;
        this->recall *= other.recall;
        this->f1_score *= other.f1_score;
        this->mcc *= other.mcc;
        return *this;
    }

    inline Stats operator/=(const double divisor)
    {
        this->accuracy /= divisor;
        this->precision /= divisor;
        this->recall /= divisor;
        this->f1_score /= divisor;
        this->mcc /= divisor;
        return *this;
    }

    inline Stats operator/=(const Stats &other)
    {
        this->accuracy /= other.accuracy;
        this->precision /= other.precision;
        this->recall /= other.recall;
        this->f1_score /= other.f1_score;
        this->mcc /= other.mcc;
        return *this;
    }

    inline Stats sqrt()
    {
        return Stats(
            std::sqrt(this->accuracy),
            std::sqrt(this->precision),
            std::sqrt(this->recall),
            std::sqrt(this->f1_score),
            std::sqrt(this->mcc));
    }
};

inline std::ostream &operator<<(std::ostream &os, const Stats &stats)
{
    os << "Accuracy: " << stats.accuracy << std::endl;
    os << "Precision: " << stats.precision << std::endl;
    os << "Recall: " << stats.recall << std::endl;
    os << "F1 Score: " << stats.f1_score << std::endl;
    os << "MCC: " << stats.mcc;
    return os;
}

class Calculator
{
public:
    inline double calc_accuracy(std::vector<bool> &gold, std::vector<bool> &pred)
    {
        int correct = 0;
        for (int i = 0; i < gold.size(); i++)
        {
            if (gold[i] == pred[i])
                correct++;
        }
        return (double)correct / gold.size();
    }

    inline double calc_precision(std::vector<bool> &gold, std::vector<bool> &pred)
    {
        int tp = 0, fp = 0;
        for (int i = 0; i < gold.size(); i++)
        {
            if (gold[i] && pred[i])
                tp++;
            else if (pred[i] && !gold[i])
                fp++;
        }
        return (double)tp / (tp + fp);
    }

    inline double calc_recall(std::vector<bool> &gold, std::vector<bool> &pred)
    {
        int tp = 0, fn = 0;
        for (int i = 0; i < gold.size(); i++)
        {
            if (gold[i] && pred[i])
                tp++;
            else if (gold[i] && !pred[i])
                fn++;
        }
        return (double)tp / (tp + fn);
    }

    inline double calc_f1_score(std::vector<bool> &gold, std::vector<bool> &pred)
    {
        int tp = 0, fn = 0, fp = 0;
        for (int i = 0; i < gold.size(); i++)
        {
            if (gold[i] && pred[i])
                tp++;
            else if (gold[i] && !pred[i])
                fn++;
            else if (!gold[i] && pred[i])
                fp++;
        }
        return (double)(2 * tp) / (2 * tp + fn + fp);
    }

    inline double calc_mcc(std::vector<bool> &gold, std::vector<bool> &pred)
    {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < gold.size(); i++)
        {
            if (gold[i] && pred[i])
                tp++;
            else if (!gold[i] && !pred[i])
                tn++;
            else if (gold[i] && !pred[i])
                fn++;
            else if (!gold[i] && pred[i])
                fp++;
        }
        return (double)(tp * tn - fp * fn) / std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    }

    inline double calc_mean(std::vector<double> &values)
    {
        double sum = 0;
        for (double value : values)
            sum += value;
        return sum / values.size();
    }

    inline double calc_stddev(std::vector<double> &values)
    {
        double mean = 0;
        double sum = 0;
        for (double value : values)
        {
            sum += value * value;
            mean += value;
        }
        mean /= values.size();
        return std::sqrt((sum / values.size()) - (mean * mean));
    }

    inline std::pair<double, double> calc_mean_stddev(std::vector<double> &values)
    {
        double mean = 0;
        double sum = 0;
        for (double value : values)
        {
            sum += value * value;
            mean += value;
        }
        mean /= values.size();
        double stddev = std::sqrt((sum / values.size()) - (mean * mean));
        return {mean, stddev};
    }
};