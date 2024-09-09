#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <unistd.h>

class Plotter
{
private:
    FILE *gnuplotPipe;
    int cnt_files = 0;

    /**
     * @brief Writes data to a file
     * @tparam T2
     * @param filename: name of the file
     * @param y: vector of y-axis values
     * @overload
     */
    template <typename T2>
    inline void _write_data(const std::string filename, const std::vector<T2> y)
    {
        std::ofstream fout(filename);
        for (int i = 0; i < y.size(); i++)
            fout << i << " " << y[i] << "\n";
        fout.close();
    }

    /**
     * @brief Writes data to a file
     * @tparam T1
     * @tparam T2
     * @param filename: name of the file
     * @param x: vector of x-axis values
     * @param y: vector of y-axis values
     * @overload
     */
    template <typename T1, typename T2>
    inline void _write_data(const std::string filename, const std::vector<T1> x, const std::vector<T2> y, const T1 shift = static_cast<T1>(0))
    {
        std::ofstream fout(filename);
        for (int i = 0; i < x.size(); i++)
        {
            if (i >= y.size())
                break;
            fout << x[i] + shift << " " << y[i] << "\n";
        }
        fout.close();
    }

    /**
     * @brief Writes data to a file
     * @tparam T1
     * @tparam T2
     * @param filename: name of the file
     * @param x: vector of x-axis values
     * @param y: vector of y-axis values
     * @overload
     */
    template <typename T1, typename T2>
    inline void _write_data(const std::string filename, const std::vector<T1> x, const std::vector<T2> ub, const std::vector<T2> lb)
    {
        std::ofstream fout(filename);
        for (int i = 0; i < x.size(); i++)
        {
            if (i >= ub.size() || i >= lb.size())
                break;
            fout << x[i] << " " << ub[i] << " " << lb[i] << "\n";
        }
        fout.close();
    }


public:
    enum LineStyle
    {
        POINT,             // 0
        SOLID,             // 1
        DASHED,            // 2
        DOTTED,            // 3
        DASH_N_DOT,        // 4
        DASH_N_DOUBLE_DOT, // 5
    };

    enum MarkerStyle
    {
        None,    // 0
        Plus,    // 1
        Cross,   // 2
        Star,    // 3
        Box,     // 4
        BoxF,    // 5
        Circle,  // 6
        CircleF, // 7
        TriU,    // 8
        TriUF,   // 9
        TriD,    // 10
        TriDF,   // 11
        Dia,     // 12
        DiaF,    // 13
        Pent,    // 14
        PentF,   // 15
        C0,      // 16
        C1,      // 17
        C2,      // 18
        C3,      // 19
        C4,      // 20
        C5,      // 21
        C6,      // 22
        C7,      // 23
        C8,      // 24
        C9,      // 25
        C10,     // 26
        C11,     // 27
        C12,     // 28
        C13,     // 29
        C14,     // 30
        C15,     // 31
        S0,      // 32
        S1,      // 33
        S2,      // 34
        S3,      // 35
        S4,      // 36
        S5,      // 37
        S6,      // 38
        S7,      // 39
        S8,      // 40
        S9,      // 41
        S10,     // 42
        S11,     // 43
        S12,     // 44
        S13,     // 45
        S14,     // 46
        S15,     // 47
        D0,      // 48
        D1,      // 49
        D2,      // 50
        D3,      // 51
        D4,      // 52
        D5,      // 53
        D6,      // 54
        D7,      // 55
        D8,      // 56
        D9,      // 57
        D10,     // 58
        D11,     // 59
        D12,     // 60
        D13,     // 61
        D14,     // 62
        D15,     // 63
        BoxE,    // 64
        CircW,   // 65
        TriUW,   // 66
        DiaW,    // 67
        PentW,   // 68
        CircF,   // 69
        Pls,     // 70
        Crs,     // 71
    };

    /**
     *  @brief  Constructor
     *  @param  size_x: width of the plot in pixels
     *  @param  size_y: height of the plot in pixels
     *  @param  fontSize: font size to be used in the plot
     *  @param  debugMode: if true, writes the gnuplot commands to a file instead of executing them
     */
    inline Plotter(int size_x = 1200, int size_y = 900, int fontSize = 20, bool debugMode = false)
    {
        if (debugMode)
            gnuplotPipe = fopen("debug_plotter.txt", "w");
        else
            gnuplotPipe = popen("gnuplot -persistent", "w");

        if (gnuplotPipe)
            fprintf(gnuplotPipe, "set terminal pngcairo enhanced font ',%d' size %d, %d\n", fontSize, size_x, size_y);
        else
            std::cerr << "Could not set up pipe with gnuplot" << std::endl;
    }

    /**
     *  @brief  Destructor
     */
    inline virtual ~Plotter()
    {
        if (gnuplotPipe)
        {
            fflush(gnuplotPipe);
            pclose(gnuplotPipe);

            for (int i = 0; i < cnt_files; i++)
            {
                unlink((std::to_string(i) + ".dat").c_str());
                // std::remove((std::to_string(i) + ".dat").c_str());
            }
        }
    }

    /**
     *  @brief  Resets gnuplot settings to default
     *  @param  size_x: width of the plot in pixels
     *  @param  size_y: height of the plot in pixels
     *  @param  fontSize: font size to be used in the plot
     */
    inline void reset(int size_x = 1200, int size_y = 900, int fontSize = 20)
    {
        fflush(gnuplotPipe);
        fprintf(gnuplotPipe, "\nreset\n");
        fprintf(gnuplotPipe, "set terminal pngcairo enhanced font ',%d' size %d, %d\n", fontSize, size_x, size_y);
    }

    /**
     *  @brief  Sends an empty command to gnuplot, used to flush commands
     */
    inline void plot()
    {
        if (gnuplotPipe)
        {
            fprintf(gnuplotPipe, "\n");
            fflush(gnuplotPipe);
        }
    }

    /**
     * @brief  Sets multiplot layout
     * @param  multi_layout_x: number of plots in each row
     * @param  multi_layout_y: number of plots in each column
     * @param  title: title of the multiplot
     * @note  `title` is not a string, it is a char array; use string.c_str() to convert a string to char array
     */
    inline void set_multiplot(int multi_layout_x = 3, int multi_layout_y = 4, const char *title = "")
    {
        fprintf(gnuplotPipe, "set multiplot layout %d, %d title '%s'\n", multi_layout_x, multi_layout_y, title);
    }

    /**
     * @brief  Unsets multiplot layout
     */
    inline void unset_multiplot()
    {
        fprintf(gnuplotPipe, "unset multiplot\n");
    }

    /**
     * @brief Sets x-axis label
     * @param label: label text for the x-axis
     * @note  `label` is not a string, it is a char array; use string.c_str() to convert a string to char array
     */
    void set_xlabel(const char *label)
    {
        if (gnuplotPipe)
            fprintf(gnuplotPipe, "\nset xlabel '%s'\n", label);
    }

    /**
     * @brief Sets y-axis label
     * @param label: label text for the y-axis
     * @note  `label` is not a string, it is a char array; use string.c_str() to convert a string to char array
     */
    void set_ylabel(const char *label)
    {
        if (gnuplotPipe)
            fprintf(gnuplotPipe, "\nset ylabel '%s'\n", label);
    }

    /**
     * @brief Sets plot title
     * @param title: title text for the plot
     * @note  `title` is not a string, it is a char array; use string.c_str() to convert a string to char array
     */
    void set_title(const char *title)
    {
        if (gnuplotPipe)
            fprintf(gnuplotPipe, "\nset title '%s'\n", title);
    }

    /**
     * @brief Sets output plot path
     * @param savePath: path to save the plot
     * @note  `savepath` is not a string, it is a char array; use string.c_str() to convert a string to char array
     */
    void set_savePath(const char *savePath)
    {
        if (gnuplotPipe)
            fprintf(gnuplotPipe, "\nset output '%s'\n", savePath);
    }

    /**
     * @brief Enables or disables grid
     * @param show_grid: if true, shows the grid; otherwise, hides the grid
     * @note  To hide the grid, use show_grid(false)
     */
    inline void show_grid(bool show_grid = true)
    {
        if (gnuplotPipe)
        {
            if (show_grid)
                fprintf(gnuplotPipe, "set grid\n");
            else
                fprintf(gnuplotPipe, "unset grid\n");
        }
    }

    /**
     * @brief Sets the position of the legend
     * @param position: position of the legend (e.g., "right", "left", "top", "bottom")
     * @note  1. To hide the legend, use unset_legend()
     * @note  2. `position` is not a string, it is a char array; use string.c_str() to convert a string to char array
     */
    inline void set_legend(const char *position = "right")
    {
        if (gnuplotPipe)
            fprintf(gnuplotPipe, "set key %s\n", position);
    }

    /**
     * @brief Hides the legend
     * @note To set the position of the legend, use set_legend()
     */
    inline void unset_legend()
    {
        if (gnuplotPipe)
            fprintf(gnuplotPipe, "unset key\n");
    }

    /**
     * @brief Sets x-axis range
     * @param min: minimum value of the x-axis
     * @param max: maximum value of the x-axis
     */
    inline void set_xlim(double min, double max)
    {
        if (gnuplotPipe)
            fprintf(gnuplotPipe, "set xrange [%f:%f]\n", min, max);
    }

    /**
     * @brief Sets y-axis range
     * @param min: minimum value of the y-axis
     * @param max: maximum value of the y-axis
     */
    inline void set_ylim(double min, double max)
    {
        if (gnuplotPipe)
            fprintf(gnuplotPipe, "set yrange [%f:%f]\n", min, max);
    }

    /**
     * @brief Sets x-axis ticks
     * @tparam T2: type of the ticks (string or char array)
     * @param ticks: vector of ticks
     * @overload
     */
    template <typename T2>
    void xticks(const std::vector<T2> &ticks)
    {
        std::string xtics_cmd = "set xtics (";
        for (int i = 0; i < ticks.size() - 1; i++)
            xtics_cmd += "\"" + ticks[i] + "\" " + std::to_string(i) + ", ";

        xtics_cmd += "\"" + ticks.back() + "\" " + std::to_string(ticks.size() - 1) + ")\n";
        fprintf(gnuplotPipe, "%s", xtics_cmd.c_str());
    }

    /**
     * @brief  Sets x-axis ticks
     * @tparam T1: type of the x-axis values
     * @tparam T2: type of the ticks (string or char array)
     * @param x: vector of x-axis values
     * @param ticks: vector of tick labels
     * @overload
     */
    template <typename T1, typename T2>
    void xticks(const std::vector<T1> &x, const std::vector<T2> &ticks)
    {
        if (x.size() != ticks.size())
            throw std::runtime_error("ERROR: xticks size doesn't match with x!");

        std::string xtics_cmd = "set xtics (";
        for (int i = 0; i < x.size() - 1; i++)
            xtics_cmd += "\"" + ticks[i] + "\" " + std::to_string(x[i]) + ", ";

        xtics_cmd += "\"" + ticks.back() + "\" " + std::to_string(x.back()) + ")\n";
        fprintf(gnuplotPipe, "%s", xtics_cmd.c_str());
    }

    /**
     * @brief Sets y-axis ticks
     * @tparam T2: type of the ticks (string or char array)
     * @param ticks: vector of ticks
     * @overload
     */
    template <typename T2>
    void yticks(const std::vector<T2> &ticks)
    {
        std::string ytics_cmd = "set ytics (";
        for (int i = 0; i < ticks.size() - 1; i++)
            ytics_cmd += "\"" + ticks[i] + "\" " + std::to_string(i) + ", ";

        ytics_cmd += "\"" + ticks.back() + "\" " + std::to_string(ticks.size() - 1) + ")\n";
        fprintf(gnuplotPipe, "%s", ytics_cmd.c_str());
    }

    /**
     * @brief  Sets y-axis ticks
     * @tparam T1: type of the y-axis values
     * @tparam T2: type of the ticks (string or char array)
     * @param y: vector of y-axis values
     * @param ticks: vector of tick labels
     * @overload
     */
    template <typename T1, typename T2>
    void yticks(const std::vector<T1> &y, const std::vector<T2> &ticks)
    {
        if (y.size() != ticks.size())
            throw std::runtime_error("ERROR: yticks size doesn't match with y!");

        std::string ytics_cmd = "set ytics (";
        for (int i = 0; i < y.size() - 1; i++)
            ytics_cmd += "\"" + ticks[i] + "\" " + std::to_string(y[i]) + ", ";

        ytics_cmd += "\"" + ticks.back() + "\" " + std::to_string(y.back()) + ")\n";
        fprintf(gnuplotPipe, "%s", ytics_cmd.c_str());
    }

    /**
     * @brief Creates a Box Plot
     * @param x: vector of x labels
     * @param y: vector of vectors containing the data for each box
     * @param show_xticks: if true, shows x-ticks; otherwise, hides x-ticks
     * @param box_width: width of the box
     * @param color: color of the box plot
     */
    inline void createBoxPlot(const std::vector<std::string> &x, const std::vector<std::vector<double>> &y, const bool show_xticks = true, const double box_width = 0.5, const char *color = "auto")
    {
        fprintf(gnuplotPipe, "set style data boxplot\n");
        fprintf(gnuplotPipe, "set style boxplot outliers pointtype 7\n");

        if (color != "auto")
            for (int i = 0; i < y.size(); i++)
                fprintf(gnuplotPipe, "set linetype %d lc '%s' lw 2\n", i + 1, color);

        std::string filename = std::to_string(cnt_files) + ".dat";
        std::ofstream fout(filename);
        for (int j = 0; j < y[0].size() - 1; j++)
        {
            for (int i = 0; i < y.size() - 1; i++)
                fout << y[i][j] << " ";
            fout << y.back()[j] << "\n";
        }
        fout.close();

        fprintf(gnuplotPipe, "plot '%s' using (1):1 title '' with boxplot,", filename.c_str());
        for (int i = 1; i < y.size(); i++)
            fprintf(gnuplotPipe, "'' using (%d):%d title '' with boxplot,", i + 1, i + 1);

        fprintf(gnuplotPipe, "\n");
        fprintf(gnuplotPipe, "unset style boxplot\n");
        cnt_files++;
    }

    /**
     * @brief Creates a Scatter Plot
     * @tparam T2: type of the y-axis values
     * @param y: vector of y-axis values
     * @param point_type: type of the point (e.g., "O", "X", "s", "d", "p", "h", "1", "2", etc.)
     * @param point_size: size of the point
     * @param title: title of the plot
     * @param point_color: color of the point
     * @param set_range: if true, automatically sets the axes range of the plot overriding any previous settings
     * @note `title`, `point_type` and `point_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @overload
     */
    template <typename T2>
    inline void createScatterPlot(const std::vector<T2> &y, const char *point_type = "O", const double point_size = 1.0, const char *title = "", const char *point_color = "auto", const bool set_range = false)
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, y);

        if (set_range)
        {
            fprintf(gnuplotPipe, "stats '%s' using 1:2 nooutput\n", filename.c_str());
            fprintf(gnuplotPipe, "x_offset = (STATS_max_x - STATS_min_x) * 0.05\n");
            fprintf(gnuplotPipe, "y_offset = (STATS_max_y - STATS_min_y) * 0.05\n");
            fprintf(gnuplotPipe, "set xrange [STATS_min_x - x_offset:STATS_max_x + x_offset]\n");
            fprintf(gnuplotPipe, "set yrange [STATS_min_y - y_offset:STATS_max_y + y_offset]\n");
        }

        fprintf(gnuplotPipe, "plot ");
        if (point_color == "auto")
            fprintf(gnuplotPipe, "\"%s\" using 1:2 with points pointtype '%s' pointsize %f title '%s'", filename.c_str(), point_type, point_size, title);
        else
            fprintf(gnuplotPipe, "\"%s\" using 1:2 with points pointtype '%s' pointsize %f linecolor '%s' title '%s'", filename.c_str(), point_type, point_size, point_color, title);

        cnt_files++;
    }

    /**
     * @brief Creates a Scatter Plot
     * @tparam T1: type of the x-axis values
     * @tparam T2: type of the y-axis values
     * @param x: vector of x-axis values
     * @param y: vector of y-axis values
     * @param point_type: type of the point (e.g., "O", "X", "s", "d", "p", "h", "1", "2", etc.)
     * @param point_size: size of the point
     * @param title: title of the plot
     * @param point_color: color of the point
     * @param set_range: if true, automatically sets the axes range of the plot overriding any previous settings
     * @note `title`, `point_type` and `point_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @overload
     */
    template <typename T1, typename T2>
    inline void createScatterPlot(const std::vector<T1> &x, const std::vector<T2> &y, const char *point_type = "O", const double point_size = 1.0, const char *title = "", const char *point_color = "auto", const bool set_range = false)
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, x, y, static_cast<T1>(0));

        if (set_range)
        {
            fprintf(gnuplotPipe, "stats '%s' using 1:2 nooutput\n", filename.c_str());
            fprintf(gnuplotPipe, "x_offset = (STATS_max_x - STATS_min_x) * 0.05\n");
            fprintf(gnuplotPipe, "y_offset = (STATS_max_y - STATS_min_y) * 0.05\n");
            fprintf(gnuplotPipe, "set xrange [STATS_min_x - x_offset:STATS_max_x + x_offset]\n");
            fprintf(gnuplotPipe, "set yrange [STATS_min_y - y_offset:STATS_max_y + y_offset]\n");
        }

        fprintf(gnuplotPipe, "plot ");
        if (point_color == "auto")
            fprintf(gnuplotPipe, "\"%s\" using 1:2 with points pointtype '%s' pointsize %f title '%s'", filename.c_str(), point_type, point_size, title);
        else
            fprintf(gnuplotPipe, "\"%s\" using 1:2 with points pointtype '%s' pointsize %f linecolor '%s' title '%s'", filename.c_str(), point_type, point_size, point_color, title);

        cnt_files++;
    }

    /**
     * @brief Adds a Scatter Plot to existing plot
     * @tparam T2: type of the y-axis values
     * @param y: vector of y-axis values
     * @param point_type: type of the point (e.g., "O", "X", "s", "d", "p", "h", "1", "2", etc.)
     * @param point_size: size of the point
     * @param title: title of the plot
     * @param point_color: color of the point
     * @param set_range: if true, automatically sets the axes range of the plot overriding any previous settings
     * @note `title`, `point_type` and `point_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @overload
     */
    template <typename T2>
    inline void addScatterPlot(const std::vector<T2> &y, const char *point_type = "O", const double point_size = 1.0, const char *title = "", const char *point_color = "auto")
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, y);

        if (point_color == "auto")
            fprintf(gnuplotPipe, ", \"%s\" using 1:2 with points pointtype '%s' pointsize %f title '%s'", filename.c_str(), point_type, point_size, title);
        else
            fprintf(gnuplotPipe, ", \"%s\" using 1:2 with points pointtype '%s' pointsize %f linecolor '%s' title '%s'", filename.c_str(), point_type, point_size, point_color, title);

        cnt_files++;
    }

    /**
     * @brief Adds a Scatter Plot to existing plot
     * @tparam T1: type of the x-axis values
     * @tparam T2: type of the y-axis values
     * @param x: vector of x-axis values
     * @param y: vector of y-axis values
     * @param point_type: type of the point (e.g., "O", "X", "s", "d", "p", "h", "1", "2", etc.)
     * @param point_size: size of the point
     * @param title: title of the plot
     * @param point_color: color of the point
     * @param set_range: if true, automatically sets the axes range of the plot overriding any previous settings
     * @note `title`, `point_type` and `point_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @overload
     */
    template <typename T1, typename T2>
    inline void addScatterPlot(const std::vector<T1> &x, const std::vector<T2> &y, const char *point_type = "O", const double point_size = 1.0, const char *title = "", const char *point_color = "auto")
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, x, y, static_cast<T1>(0));

        if (point_color == "auto")
            fprintf(gnuplotPipe, ", \"%s\" using 1:2 with points pointtype '%s' pointsize %f title '%s'", filename.c_str(), point_type, point_size, title);
        else
            fprintf(gnuplotPipe, ", \"%s\" using 1:2 with points pointtype '%s' pointsize %f linecolor '%s' title '%s'", filename.c_str(), point_type, point_size, point_color, title);

        cnt_files++;
    }

    /**
     * @brief Creates a Histogram
     * @tparam T2: type of the y-axis values
     * @param y: vector of y-axis values
     * @param bin_width: width of the bin; if 0, the bin width is automatically determined
     * @param color: color of the histogram bars
     * @param title: title of the plot
     * @param opacity: opacity of the histogram bars
     * @note `title` and `color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @overload
     */
    template <typename T2>
    inline void createHistogram(const std::vector<T2> &y, const double bin_width = 0.0, const char *color = "auto", const char *title = "", const double opacity = 1.0)
    {
        fprintf(gnuplotPipe, "set style data histograms\n");

        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, y);

        // Determine the range of the data
        fprintf(gnuplotPipe, "stats '%s' using 1:2 nooutput\n", filename.c_str());
        fprintf(gnuplotPipe, "y_offset = (STATS_max_y - STATS_min_y) * 0.05\n");
        fprintf(gnuplotPipe, "set yrange [STATS_min_y - y_offset:STATS_max_y + y_offset]\n");

        fprintf(gnuplotPipe, "set boxwidth %f relative\n", bin_width);
        fprintf(gnuplotPipe, "set style fill solid %f\n", opacity);

        if (std::string(color) == "auto")
            fprintf(gnuplotPipe, "plot '%s' using 2 title '%s'\n", filename.c_str(), title);
        else
            fprintf(gnuplotPipe, "plot '%s' using 2 line_color '%s' title '%s'\n", filename.c_str(), color, title);

        cnt_files++;
    }

    /**
     * @brief Creates a Histogram
     * @tparam T1: type of the x-axis values
     * @tparam T2: type of the y-axis values
     * @param x: vector of x-axis values
     * @param y: vector of y-axis values
     * @param bin_width: width of the bin; if 0, the bin width is automatically determined
     * @param color: color of the histogram bars
     * @param title: title of the plot
     * @param opacity: opacity of the histogram bars
     * @note `title` and `color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @overload
     */
    template <typename T1, typename T2>
    inline void createHistogram(const std::vector<T1> &x, const std::vector<T2> &y, const double bin_width = 0.0, const char *color = "auto", const char *title = "", const double opacity = 1.0)
    {
        fprintf(gnuplotPipe, "set style data histograms\n");

        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, x, y, static_cast<T1>(0));

        // Determine the range of the data
        fprintf(gnuplotPipe, "stats '%s' using 1:2 nooutput\n", filename.c_str());
        fprintf(gnuplotPipe, "y_offset = (STATS_max_y - STATS_min_y) * 0.05\n");
        fprintf(gnuplotPipe, "set yrange [STATS_min_y - y_offset:STATS_max_y + y_offset]\n");

        fprintf(gnuplotPipe, "set boxwidth %f relative\n", bin_width);
        fprintf(gnuplotPipe, "set style fill solid %f\n", opacity);

        if (std::string(color) == "auto")
            fprintf(gnuplotPipe, "plot '%s' using 2:xtic(1) title '%s'\n", filename.c_str(), title);
        else
            fprintf(gnuplotPipe, "plot '%s' using 2:xtic(1) line_color '%s' title '%s'\n", filename.c_str(), color, title);

        cnt_files++;
    }

    /**
     * @brief Creates a Line Plot
     * @tparam T2: type of the y-axis values
     * @param y: vector of y-axis values
     * @param line_title: title of the line plot
     * @param line_color: color of the line plot
     * @param marker: point marker style
     * @param set_range: if true, automatically sets the axes range of the plot overriding any previous settings
     * @note 1. `line_title` and `line_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @note 2. turn `set_range` to false when plotting multiple lines in the plot using add_plot and rather set the limits manually.
     * @overload
     */
    template <typename T2>
    inline void createPlot(const std::vector<T2> &y, const char *line_title = "", const char *line_color = "auto", const MarkerStyle marker = None, const double point_size = 1.0, const double line_width = 1.0, const LineStyle line_style = SOLID, const bool set_range = false)
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, y);

        if (set_range)
        {
            fprintf(gnuplotPipe, "stats '%s' using 1:2 nooutput\n", filename.c_str());
            fprintf(gnuplotPipe, "x_offset = (STATS_max_x - STATS_min_x) * 0.05\n");
            fprintf(gnuplotPipe, "y_offset = (STATS_max_y - STATS_min_y) * 0.05\n");
            fprintf(gnuplotPipe, "set xrange [STATS_min_x - x_offset:STATS_max_x + x_offset]\n");
            fprintf(gnuplotPipe, "set yrange [STATS_min_y - y_offset:STATS_max_y + y_offset]\n");
        }
        fprintf(gnuplotPipe, "plot ");
        if (line_color == "auto")
            fprintf(gnuplotPipe, "\"%s\" using 1:2 smooth unique with linespoints pointtype %d pointsize %f dashtype %d linewidth %f title '%s'", filename.c_str(), marker, point_size, line_style, line_width, line_title);
        else
            fprintf(gnuplotPipe, "\"%s\" using 1:2 smooth unique with linespoints pointtype %d pointsize %f dashtype %d linewidth %f linecolor '%s' title '%s'", filename.c_str(), marker, point_size, line_style, line_width, line_color, line_title);

        cnt_files++;
    }

    /**
     * @brief Creates a Line Plot
     * @tparam T1: type of the x-axis values
     * @tparam T2: type of the y-axis values
     * @param x: vector of x-axis values
     * @param y: vector of y-axis values
     * @param line_title: title of the line plot
     * @param line_color: color of the line plot
     * @param shift: shift the x-axis values by a constant value
     * @param set_range: if true, automatically sets the axes range of the plot overriding any previous settings
     * @note 1. `line_title` and `line_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @note 2. turn `set_range` to false when plotting multiple lines in the plot using add_plot and rather set the limits manually.
     * @overload
     */
    template <typename T1, typename T2>
    inline void createPlot(const std::vector<T1> &x, const std::vector<T2> &y, const char *line_title = "", const char *line_color = "auto", const MarkerStyle marker = None, const double point_size = 1.0, const double line_width = 1.0, const LineStyle line_style = SOLID, const T1 shift = static_cast<T1>(0), const bool set_range = false)
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, x, y, shift);

        if (set_range)
        {
            fprintf(gnuplotPipe, "stats '%s' using 1:2 nooutput\n", filename.c_str());
            fprintf(gnuplotPipe, "x_offset = (STATS_max_x - STATS_min_x) * 0.05\n");
            fprintf(gnuplotPipe, "y_offset = (STATS_max_y - STATS_min_y) * 0.05\n");
            fprintf(gnuplotPipe, "set xrange [STATS_min_x - x_offset:STATS_max_x + x_offset]\n");
            fprintf(gnuplotPipe, "set yrange [STATS_min_y - y_offset:STATS_max_y + y_offset]\n");
        }

        fprintf(gnuplotPipe, "plot ");
        if (line_color == "auto")
            fprintf(gnuplotPipe, "\"%s\" using 1:2 smooth unique with linespoints pointtype %d pointsize %f dashtype %d linewidth %f title '%s'", filename.c_str(), marker, point_size, line_style, line_width, line_title);
        else
            fprintf(gnuplotPipe, "\"%s\" using 1:2 smooth unique with linespoints pointtype %d pointsize %f dashtype %d linewidth %f linecolor '%s' title '%s'", filename.c_str(), marker, point_size, line_style, line_width, line_color, line_title);

        cnt_files++;
    }

    /**
     * @brief Creates a Line Plot
     * @tparam T2: type of the y-axis values
     * @param y: vector of y-axis values
     * @param line_title: title of the line plot
     * @param line_color: color of the line plot
     * @note 1. `line_title` and `line_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @note 2. `line_style` is an enum; use `Plotter::SOLID`, `Plotter::DASHED`, `Plotter::DOTTED`, `Plotter::DASH_N_DOT`, `Plotter::DASH_N_DOUBLE_DOT` to set the line style
     * @overload
     */
    template <typename T2>
    inline void addPlot(const std::vector<T2> &y, const char *line_title = "", const char *line_color = "auto", const MarkerStyle marker = None, const double point_size = 1.0, const double line_width = 1.0, const LineStyle line_style = SOLID)
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, y);

        if (line_color == "auto")
            fprintf(gnuplotPipe, ", \"%s\" using 1:2 smooth unique with linespoints pointtype %d pointsize %f dashtype %d linewidth %f title '%s'", filename.c_str(), marker, point_size, line_style, line_width, line_title);
        else
            fprintf(gnuplotPipe, ", \"%s\" using 1:2 smooth unique with linespoints pointtype %d pointsize %f dashtype %d linewidth %f linecolor '%s' title '%s'", filename.c_str(), marker, point_size, line_style, line_width, line_color, line_title);

        cnt_files++;
    }

    /**
     * @brief Creates a Line Plot
     * @tparam T1: type of the x-axis values
     * @tparam T2: type of the y-axis values
     * @param x: vector of x-axis values
     * @param y: vector of y-axis values
     * @param line_title: title of the line plot
     * @param line_color: color of the line plot
     * @param shift: shift the x-axis values by a constant value
     * @param set_range: if true, automatically sets the axes range of the plot overriding any previous settings
     * @note `line_title` and `line_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @overload
     */
    template <typename T1, typename T2>
    inline void addPlot(const std::vector<T1> &x, const std::vector<T2> &y, const char *line_title = "", const char *line_color = "auto", const MarkerStyle marker = None, const double point_size = 1.0, const double line_width = 1.0, const LineStyle line_style = SOLID, const T1 shift = static_cast<T1>(0))
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, x, y, shift);

        if (line_color == "auto")
            fprintf(gnuplotPipe, ", \"%s\" using 1:2 smooth unique with linespoints pointtype %d pointsize %f dashtype %d linewidth %f title '%s'", filename.c_str(), marker, point_size, line_style, line_width, line_title);
        else
            fprintf(gnuplotPipe, ", \"%s\" using 1:2 smooth unique with linespoints pointtype %d pointsize %f dashtype %d linewidth %f linecolor '%s' title '%s'", filename.c_str(), marker, point_size, line_style, line_width, line_color, line_title);

        cnt_files++;
    }

    /**
     * @brief Creates a Line Plot
     * @tparam T2: type of the y-axis values
     * @param ub: vector of upper bound of y-axis values
     * @param lb: vector of lower bound of y-axis values
     * @param color: color of the line plot
     * @note 1. `line_title` and `line_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @note 2. `line_style` is an enum; use `Plotter::SOLID`, `Plotter::DASHED`, `Plotter::DOTTED`, `Plotter::DASH_N_DOT`, `Plotter::DASH_N_DOUBLE_DOT` to set the line style
     * @overload
     */
    template <typename T2>
    inline void fillBetween(const std::vector<T2> &ub, const std::vector<T2> &lb, const char *color = "auto", const double alpha = 0.2)
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        std::vector<int> x(ub.size());
        for (int i = 0; i < ub.size(); i++)
            x[i] = i;
        _write_data(filename, x, ub, lb);

        if (color == "auto")
            fprintf(gnuplotPipe, ", \"%s\" using 1:2:3 with filledcurves fill transparent solid %f title ''", filename.c_str(), alpha);
        else
            fprintf(gnuplotPipe, ", \"%s\" using 1:2:3 with filledcurves linecolor '%s' fill transparent solid %f title ''", filename.c_str(), color, alpha);

        cnt_files++;
    }

    /**
     * @brief Creates a Line Plot
     * @tparam T2: type of the y-axis values
     * @param x: vector of x-axis values
     * @param ub: vector of upper bound of y-axis values
     * @param lb: vector of lower bound of y-axis values
     * @param color: color of the line plot
     * @note 1. `line_title` and `line_color` are not strings, they are char arrays; use string.c_str() to convert a string to char array
     * @note 2. `line_style` is an enum; use `Plotter::SOLID`, `Plotter::DASHED`, `Plotter::DOTTED`, `Plotter::DASH_N_DOT`, `Plotter::DASH_N_DOUBLE_DOT` to set the line style
     * @overload
     */
    template <typename T1, typename T2>
    inline void fillBetween(const std::vector<T1> &x, const std::vector<T2> &ub, const std::vector<T2> &lb, const char *color = "auto", const double alpha = 0.2)
    {
        std::string filename = std::to_string(cnt_files) + ".dat";
        _write_data(filename, x, ub, lb);

        if (color == "auto")
            fprintf(gnuplotPipe, ", \"%s\" using 1:2:3 with filledcurves fill transparent solid %f title ''", filename.c_str(), alpha);
        else
            fprintf(gnuplotPipe, ", \"%s\" using 1:2:3 with filledcurves linecolor '%s' fill transparent solid %f title ''", filename.c_str(), color, alpha);

        cnt_files++;
    }
};
