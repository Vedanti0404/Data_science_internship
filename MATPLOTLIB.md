### Matplotlib Revision Notes

**Matplotlib Overview**
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is built on top of NumPy and provides a range of functionalities to create various plots and graphs.

**Why Use Matplotlib?**
- Built on NumPy arrays.
- Enables creation of both basic and advanced graphs.
- Provides an easy-to-use interface for plotting.

**Basic Plotting**
1. **Creating a Basic Plot:**
   ```python
   plt.plot()
   ```
   An empty plot is created with just the figure and axes.

2. **Suppressing Output:**
   ```python
   plt.plot();
   ```

3. **Displaying the Plot:**
   ```python
   plt.plot()
   plt.show()
   ```

4. **Plotting Data Points:**
   ```python
   study_time = [1,2,3,4,5,6,6.5]
   marks_obtained = [10,15,23,30,36,40,45]
   plt.plot(study_time, marks_obtained)
   plt.show()
   ```

5. **Plotting Random Data:**
   ```python
   x = np.random.randint(3,10, size=(4,3))
   y = [1,2,3,4]
   plt.plot(x, y)
   plt.show()
   ```

**Object-Oriented API vs. Pyplot API**
The object-oriented API offers better control and flexibility for managing figures and axes.

1. **Method 1:**
   ```python
   figure = plt.figure()
   axis = figure.add_subplot()
   axis.plot(study_time, marks_obtained)
   plt.show()
   ```

2. **Method 2:**
   ```python
   figure = plt.figure()
   axis = figure.add_axes([1,1,1,1], "Rohit")  # (left, bottom, width, height)
   axis.plot(study_time, marks_obtained)
   plt.show()
   ```

3. **Method 3 (Recommended):**
   ```python
   figure, axis = plt.subplots()
   axis.plot(study_time, marks_obtained)
   plt.show()
   ```

**Matplotlib Workflow**
1. **Import Matplotlib:**
   ```python
   %matplotlib inline
   import matplotlib.pyplot as plt
   ```

2. **Organize Data:**
   ```python
   study_time = [1,2,3,4,5,6,6.5]
   marks_obtained = [10,15,23,30,36,40,45]
   ```

3. **Setup Plot:**
   ```python
   figure, axis = plt.subplots(figsize=(7,7))
   ```

4. **Plot Data:**
   ```python
   axis.plot(study_time, marks_obtained)
   ```

5. **Customize Plot:**
   ```python
   axis.set(title="Study Marks Ratio", xlabel="Study time", ylabel="Marks obtained")
   ```

6. **Save Plot:**
   ```python
   figure.savefig("../graphs_and_images/study-marks-ratio.png")
   ```

**Creating Various Plots**

1. **Line Plot:**
   ```python
   basic_array_x = np.array([3,14,25,28,45,50])
   basic_array_y = np.array([10,46,40,49,43,50])
   figure, axis = plt.subplots(figsize=(7,7))
   axis.plot(basic_array_x, basic_array_y)
   axis.set(title="Line Plot", xlabel="X-axis data", ylabel="Y-axis data")
   ```

2. **Scatter Plot:**
   ```python
   figure, axis = plt.subplots()
   axis.scatter(basic_array_x, basic_array_y)
   axis.set(title="Scatter Plot", xlabel="X-axis data", ylabel="Y-axis data")
   ```

3. **Scatter Plot for Exponential Values:**
   ```python
   import math
   a1 = list(range(2,25))
   a2 = [math.exp(num) for num in a1]
   figure, axis = plt.subplots()
   axis.scatter(a1, a2)
   axis.set(title="Exponential Scatter Plot", xlabel="X-axis data", ylabel="Y-axis data")
   ```

4. **Bar Graph:**
   ```python
   dict_data = {"rohit": 38, "amaya": 48, "kshitij": 11}
   figure, axis = plt.subplots()
   axis.bar(dict_data.keys(), dict_data.values())
   axis.set(title="Bar Graph", xlabel="Keys", ylabel="Values")
   ```

5. **Horizontal Bar Graph:**
   ```python
   figure, axis = plt.subplots()
   axis.barh(a1, a2)
   axis.set(title="Horizontal Bar Graph", xlabel="X-axis data", ylabel="Y-axis data")
   ```

6. **Histogram:**
   ```python
   figure, axis = plt.subplots()
   axis.hist(basic_array_x, bins=5)
   axis.set(title="Histogram Plot", xlabel="X-axis data", ylabel="Frequency")
   ```

**Subplots**
1. **Creating Multiple Subplots:**
   ```python
   figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
   ax1.plot(basic_array_x, basic_array_y)
   ax2.scatter(a1, np.exp(a1))
   ax3.bar(dict_data.keys(), dict_data.values())
   ax4.hist(np.random.randint(1,1000, size=(1000)))
   ```

2. **Accessing Subplots by Index:**
   ```python
   figure, axis = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
   axis[0, 0].plot(basic_array_x, basic_array_y)
   axis[0, 1].scatter(a1, np.exp(a1))
   axis[1, 0].bar(dict_data.keys(), dict_data.values())
   axis[1, 1].hist(np.random.randint(1,1000, size=(1000)))
   ```

**Plotting from Pandas DataFrames**
1. **Read CSV File and Fill Missing Values:**
   ```python
   file = pd.read_csv("../pandas/class-grades.csv")
   file = file.fillna(file.mean())
   ```

2. **Plotting Histograms and Various Plots:**
   ```python
   file.hist("Tutorial")
   file.plot("Roll no", "Final", title="Final Marks", xlabel="Roll no.", ylabel="Marks Obtained")
   file.plot("Roll no", "Final", title="Final Marks", xlabel="Roll no.", ylabel="Marks Obtained", kind="bar")
   file.plot("Roll no", "Final", title="Final Marks", xlabel="Roll no.", ylabel="Marks Obtained", kind="scatter")
   ```

3. **Creating Subplots from DataFrame:**
   ```python
   figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
   ax1.scatter(file["Roll no"], file["Assignment"])
   ax2.bar(file["Roll no"], file["Tutorial"])
   ax3.bar(file["Roll no"], file["Final"])
   ax4.bar(file["Roll no"], file["Midterm"])
   ```

4. **Plotting with Specific Conditions:**
   ```python
   over_50_marks = file[file["Final"] >= 50]
   over_50_marks.plot(kind="bar", x="Roll no", y="Final", title="Final Marks")
   over_50_marks.plot(kind="scatter", x="Roll no", y="Final", title="Final Marks", c="Final")
   ```

**Adding Style and Customization**
1. **Applying Styles:**
   ```python
   plt.style.use('seaborn')
   ```

2. **Saving Figures:**
   ```python
   fig.savefig("../graphs_and_images/Class-grades.png")
   ```

**Key Points:**
- Use Matplotlib for visualizing data with plots and graphs.
- Prefer the object-oriented API for better control and flexibility.
- Customize plots with titles, labels, and styles for clear data representation.
