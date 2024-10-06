#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

struct Stats
{
    double true_pos;
    double true_neg;
    double false_pos;
    double false_neg;
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    double mcc;

    inline Stats()
    {
        this->true_pos = 0;
        this->true_neg = 0;
        this->false_pos = 0;
        this->false_neg = 0;
        this->accuracy = 0;
        this->precision = 0;
        this->recall = 0;
        this->f1_score = 0;
        this->mcc = 0;
    }

    inline Stats(double true_pos, double true_neg, double false_pos, double false_neg, double accuracy, double precision, double recall, double f1_score, double mcc)
    {
        this->true_pos = true_pos;
        this->true_neg = true_neg;
        this->false_pos = false_pos;
        this->false_neg = false_neg;
        this->accuracy = accuracy;
        this->precision = precision;
        this->recall = recall;
        this->f1_score = f1_score;
        this->mcc = mcc;
    }

    inline bool is_valid()
    {
        return this->true_pos >= 0
            && this->true_neg >= 0
            && this->false_pos >= 0
            && this->false_neg >= 0
            && this->accuracy >= 0
            && this->precision >= 0
            && this->recall >= 0
            && this->f1_score >= 0
            && this->mcc >= -1
            && this->accuracy <= 1
            && this->precision <= 1
            && this->recall <= 1
            && this->f1_score <= 1
            && this->mcc <= 1
            && !std::isnan(this->true_pos)
            && !std::isnan(this->true_neg)
            && !std::isnan(this->false_pos)
            && !std::isnan(this->false_neg)
            && !std::isnan(this->accuracy)
            && !std::isnan(this->precision)
            && !std::isnan(this->recall)
            && !std::isnan(this->f1_score)
            && !std::isnan(this->mcc);
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
        this->true_pos = tp;
        this->true_neg = tn;
        this->false_pos = fp;
        this->false_neg = fn;
        this->accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
        this->precision = (double)tp / (tp + fp);
        this->recall = (double)tp / (tp + fn);
        this->f1_score = (double)(2 * tp) / (2 * tp + fn + fp);
        this->mcc = (double)(tp * tn - fp * fn) / std::sqrt((tp + fp + 0.01) * (tp + fn + 0.01) * (tn + fp + 0.01) * (tn + fn + 0.01));
        return *this;
    }

    inline bool operator==(const Stats &other) const
    {
        return this->true_pos == other.true_pos
            && this->true_neg == other.true_neg
            && this->false_pos == other.false_pos
            && this->false_neg == other.false_neg
            && this->accuracy == other.accuracy
            && this->precision == other.precision
            && this->recall == other.recall
            && this->f1_score == other.f1_score
            && this->mcc == other.mcc;
    }

    inline bool operator==(const double other) const
    {
        return this->true_pos == other
            && this->true_neg == other
            && this->false_pos == other
            && this->false_neg == other
            && this->accuracy == other
            && this->precision == other
            && this->recall == other
            && this->f1_score == other
            && this->mcc == other;
    }

    inline Stats operator++()
    {
        this->true_pos++;
        this->true_neg++;
        this->false_pos++;
        this->false_neg++;
        this->accuracy++;
        this->precision++;
        this->recall++;
        this->f1_score++;
        this->mcc++;
        return *this;
    }

    inline Stats operator--()
    {
        this->true_pos--;
        this->true_neg--;
        this->false_pos--;
        this->false_neg--;
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
            -this->true_pos,
            -this->true_neg,
            -this->false_pos,
            -this->false_neg,
            -this->accuracy,
            -this->precision,
            -this->recall,
            -this->f1_score,
            -this->mcc);
    }

    inline Stats operator+(const double increment) const
    {
        return Stats(
            this->true_pos + increment,
            this->true_neg + increment,
            this->false_pos + increment,
            this->false_neg + increment,
            this->accuracy + increment,
            this->precision + increment,
            this->recall + increment,
            this->f1_score + increment,
            this->mcc + increment);
    }

    inline Stats operator+(const Stats &other) const
    {
        return Stats(
            this->true_pos + other.true_pos,
            this->true_neg + other.true_neg,
            this->false_pos + other.false_pos,
            this->false_neg + other.false_neg,
            this->accuracy + other.accuracy,
            this->precision + other.precision,
            this->recall + other.recall,
            this->f1_score + other.f1_score,
            this->mcc + other.mcc);
    }

    inline Stats operator-(const double decrement) const
    {
        return Stats(
            this->true_pos - decrement,
            this->true_neg - decrement,
            this->false_pos - decrement,
            this->false_neg - decrement,
            this->accuracy - decrement,
            this->precision - decrement,
            this->recall - decrement,
            this->f1_score - decrement,
            this->mcc - decrement);
    }

    inline Stats operator-(const Stats &other) const
    {
        return Stats(
            this->true_pos - other.true_pos,
            this->true_neg - other.true_neg,
            this->false_pos - other.false_pos,
            this->false_neg - other.false_neg,
            this->accuracy - other.accuracy,
            this->precision - other.precision,
            this->recall - other.recall,
            this->f1_score - other.f1_score,
            this->mcc - other.mcc);
    }

    inline Stats operator*(const double multiplier) const
    {
        return Stats(
            this->true_pos * multiplier,
            this->true_neg * multiplier,
            this->false_pos * multiplier,
            this->false_neg * multiplier,
            this->accuracy * multiplier,
            this->precision * multiplier,
            this->recall * multiplier,
            this->f1_score * multiplier,
            this->mcc * multiplier);
    }

    inline Stats operator*(const Stats &other) const
    {
        return Stats(
            this->true_pos * other.true_pos,
            this->true_neg * other.true_neg,
            this->false_pos * other.false_pos,
            this->false_neg * other.false_neg,
            this->accuracy * other.accuracy,
            this->precision * other.precision,
            this->recall * other.recall,
            this->f1_score * other.f1_score,
            this->mcc * other.mcc);
    }

    inline Stats operator/(const double divisor) const
    {
        return Stats(
            this->true_pos / divisor,
            this->true_neg / divisor,
            this->false_pos / divisor,
            this->false_neg / divisor,
            this->accuracy / divisor,
            this->precision / divisor,
            this->recall / divisor,
            this->f1_score / divisor,
            this->mcc / divisor);
    }

    inline Stats operator/(const Stats &other) const
    {
        return Stats(
            this->true_pos / other.true_pos,
            this->true_neg / other.true_neg,
            this->false_pos / other.false_pos,
            this->false_neg / other.false_neg,
            this->accuracy / other.accuracy,
            this->precision / other.precision,
            this->recall / other.recall,
            this->f1_score / other.f1_score,
            this->mcc / other.mcc);
    }

    inline Stats operator+=(const double increment)
    {
        this->true_pos += increment;
        this->true_neg += increment;
        this->false_pos += increment;
        this->false_neg += increment;
        this->accuracy += increment;
        this->precision += increment;
        this->recall += increment;
        this->f1_score += increment;
        this->mcc += increment;
        return *this;
    }

    inline Stats operator+=(const Stats &other)
    {
        this->true_pos += other.true_pos;
        this->true_neg += other.true_neg;
        this->false_pos += other.false_pos;
        this->false_neg += other.false_neg;
        this->accuracy += other.accuracy;
        this->precision += other.precision;
        this->recall += other.recall;
        this->f1_score += other.f1_score;
        this->mcc += other.mcc;
        return *this;
    }

    inline Stats operator-=(const double decrement)
    {
        this->true_pos -= decrement;
        this->true_neg -= decrement;
        this->false_pos -= decrement;
        this->false_neg -= decrement;
        this->accuracy -= decrement;
        this->precision -= decrement;
        this->recall -= decrement;
        this->f1_score -= decrement;
        this->mcc -= decrement;
        return *this;
    }

    inline Stats operator-=(const Stats &other)
    {
        this->true_pos -= other.true_pos;
        this->true_neg -= other.true_neg;
        this->false_pos -= other.false_pos;
        this->false_neg -= other.false_neg;
        this->accuracy -= other.accuracy;
        this->precision -= other.precision;
        this->recall -= other.recall;
        this->f1_score -= other.f1_score;
        this->mcc -= other.mcc;
        return *this;
    }

    inline Stats operator*=(const double multiplier)
    {
        this->true_pos *= multiplier;
        this->true_neg *= multiplier;
        this->false_pos *= multiplier;
        this->false_neg *= multiplier;
        this->accuracy *= multiplier;
        this->precision *= multiplier;
        this->recall *= multiplier;
        this->f1_score *= multiplier;
        this->mcc *= multiplier;
        return *this;
    }

    inline Stats operator*=(const Stats &other)
    {
        this->true_pos *= other.true_pos;
        this->true_neg *= other.true_neg;
        this->false_pos *= other.false_pos;
        this->false_neg *= other.false_neg;
        this->accuracy *= other.accuracy;
        this->precision *= other.precision;
        this->recall *= other.recall;
        this->f1_score *= other.f1_score;
        this->mcc *= other.mcc;
        return *this;
    }

    inline Stats operator/=(const double divisor)
    {
        this->true_pos /= divisor;
        this->true_neg /= divisor;
        this->false_pos /= divisor;
        this->false_neg /= divisor;
        this->accuracy /= divisor;
        this->precision /= divisor;
        this->recall /= divisor;
        this->f1_score /= divisor;
        this->mcc /= divisor;
        return *this;
    }

    inline Stats operator/=(const Stats &other)
    {
        this->true_pos /= other.true_pos;
        this->true_neg /= other.true_neg;
        this->false_pos /= other.false_pos;
        this->false_neg /= other.false_neg;
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
            std::sqrt(this->true_pos),
            std::sqrt(this->true_neg),
            std::sqrt(this->false_pos),
            std::sqrt(this->false_neg),
            std::sqrt(this->accuracy),
            std::sqrt(this->precision),
            std::sqrt(this->recall),
            std::sqrt(this->f1_score),
            std::sqrt(this->mcc));
    }
};

inline std::ostream &operator<<(std::ostream &os, const Stats &stats)
{
    os << std::fixed << std::setprecision(3);
    os << "True Positives   : " << stats.true_pos << std::endl;
    os << "True Negatives   : " << stats.true_neg << std::endl;
    os << "False Positives  : " << stats.false_pos << std::endl;
    os << "False Negatives  : " << stats.false_neg << std::endl;
    os << "Accuracy         : " << stats.accuracy << std::endl;
    os << "Precision        : " << stats.precision << std::endl;
    os << "Recall           : " << stats.recall << std::endl;
    os << "F1 Score         : " << stats.f1_score << std::endl;
    os << "MCC              : " << stats.mcc;
    return os;
}

inline std::ostream &operator<<(std::ostream &os, const std::pair<Stats, Stats> &stats)
{
    os << std::fixed << std::setprecision(3);
    os << "True Positives   : " << stats.first.true_pos << "\t\u00B1\t" << stats.second.true_pos << std::endl;
    os << "True Negatives   : " << stats.first.true_neg << "\t\u00B1\t" << stats.second.true_neg << std::endl;
    os << "False Positives  : " << stats.first.false_pos << "\t\u00B1\t" << stats.second.false_pos << std::endl;
    os << "False Negatives  : " << stats.first.false_neg << "\t\u00B1\t" << stats.second.false_neg << std::endl;
    os << "Accuracy         : " << stats.first.accuracy << "\t\u00B1\t" << stats.second.accuracy << std::endl;
    os << "Precision        : " << stats.first.precision << "\t\u00B1\t" << stats.second.precision << std::endl;
    os << "Recall           : " << stats.first.recall << "\t\u00B1\t" << stats.second.recall << std::endl;
    os << "F1 Score         : " << stats.first.f1_score << "\t\u00B1\t" << stats.second.f1_score << std::endl;
    os << "MCC              : " << stats.first.mcc << "\t\u00B1\t" << stats.second.mcc;
    return os;
}

class Calculator
{
public:
    static inline double calc_accuracy(std::vector<bool> &gold, std::vector<bool> &pred)
    {
        int correct = 0;
        for (int i = 0; i < gold.size(); i++)
        {
            if (gold[i] == pred[i])
                correct++;
        }
        return (double)correct / gold.size();
    }

    static inline double calc_precision(std::vector<bool> &gold, std::vector<bool> &pred)
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

    static inline double calc_recall(std::vector<bool> &gold, std::vector<bool> &pred)
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

    static inline double calc_f1_score(std::vector<bool> &gold, std::vector<bool> &pred)
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

    static inline double calc_mcc(std::vector<bool> &gold, std::vector<bool> &pred)
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

    static inline double calc_mean(std::vector<double> &values)
    {
        double sum = 0;
        for (double value : values)
            sum += value;
        return sum / values.size();
    }

    static inline double calc_stddev(std::vector<double> &values)
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