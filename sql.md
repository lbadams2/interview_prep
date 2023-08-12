* `having` similar to `where` but it filters aggregated records after `group by`
```sql
select s.name as 'salesperson', count(o.number) as 'number of orders'
from salesperson s
join orders o on s.id = o.salesperson_id
group by s.name
having count(o.number)>=2;
```

* there are `intersect`, `union`, `except` keywords

* subquery, can be in select, from or where clauses
```sql
SELECT *
FROM EmployeeSalary
WHERE EmpId IN 
(SELECT EmpId from ManagerSalary);
```

* self join
```sql
SELECT DISTINCT E.FullName
FROM EmployeeDetails E
INNER JOIN EmployeeDetails M
ON E.EmpID = M.ManagerID;
```

* get duplicate records
```sql
SELECT FullName, ManagerId, DateOfJoining, City, COUNT(*)
FROM EmployeeDetails
GROUP BY FullName, ManagerId, DateOfJoining, City
HAVING COUNT(*) > 1;
```

* get nth highest value from table
```sql
select * from(
select ename, sal, dense_rank() over(order by sal desc) as r -- over is used with window functions
from Employee)
where r=&n;
```
or
```sql
select * from Employee ORDER BY `sal` DESC limit 5,1; -- this gets 6th highest, limit n-1, 1 gets nth highest
```

* case
```sql
select app_id
        , ifnull(sum(case when type = 'click' then 1 else 0 end)*1.0
        / sum(case when type = 'impression' then 1 else 0 end), 0 )AS 'CTR(click through rate)'
from dialoglog
group by app_id;
```

* find users active every day over last week
```sql
select a.login_time, count(distinct a.user_id) from 
login_info a
Left join login_info b
on a.user_id = b.user_id
where a.login_time = b.login_time - interval 1 day and a.login_time >= date_sub(curdate(), interval 7 day)
group by 1;
```

* select all fields from row with most recent (or max, min, etc) time for each user
```sql
select * from login_info
where login_time in (select max(login_time) from login_info
group by user_id)
order by login_time desc limit 1;
```

## CTEs
Date of each branch's largest ticket sold and the amount of the ticket
```sql
WITH tickets AS (
  SELECT distinct
    branch,
    date,
    unit_price * quantity AS ticket_amount,
    ROW_NUMBER() OVER ( -- creates a column that ranks the sales for each branch per day, another window function
      PARTITION BY branch
      ORDER by unit_price * quantity DESC
    ) AS position
  FROM sales
  ORDER BY 3 DESC
)
SELECT -- selects the branch/day combo with highest sales (position)
  branch,
  date,
  ticket_amount
FROM tickets
WHERE position =1
```

Total monthly revenue in London in 2021 and revenue for each branch in London
```sql
WITH london1_monthly_revenue AS (
  SELECT
    EXTRACT(MONTH FROM date) as month,
    SUM(unit_price * quantity) AS revenue
  FROM sales
  WHERE EXTRACT(YEAR FROM date) = 2021
    AND branch = 'London-1'
  GROUP BY 1
), -- have to chain together CTEs
london2_monthly_revenue AS (
  SELECT
    EXTRACT(MONTH FROM date) as month,
    SUM(unit_price * quantity) AS revenue
  FROM sales
  WHERE EXTRACT(YEAR FROM date) = 2021
    AND branch = 'London-2'
  GROUP BY 1
)
SELECT -- selects a row for each month, there is no explicit join keyword, this syntax is implicit join which is identical
  l1.month,
  l1.revenue + l2.revenue AS london_revenue,
  l1.revenue AS london1_revenue,
  l2.revenue AS london2_revenue
FROM london1_monthly_revenue l1, london2_monthly_revenue l2
WHERE l1.month = l2.month
```

Report for items sold over $90 and quantity of them sold for London-2 branch
```sql
WITH over_90_items AS (
  SELECT DISTINCT
    item,
    unit_price
  FROM sales
  WHERE unit_price >=90
),
london2_over_90 AS (
  SELECT
    o90.item,
    o90.unit_price,
    coalesce(SUM(s.quantity), 0) as total_sold -- s.quantity might be null because of left join, coalesce returns first non null value from its args
  FROM over_90_items o90
  LEFT JOIN sales s
  ON o90.item = s.item AND s.branch = 'London-2'
  GROUP BY o90.item, o90.unit_price
)
SELECT item, unit_price, total_sold
FROM   london2_over_90;
```

## Window functions
* Maintain original identity of individual rows, unlike aggregate functions
* over clause signifies a window of rows over which a window function is applied `window_func() over()`
* args to `over()` either `over(partition by col1 order by col2)` or `over(order by col2)`

* to apply the window function to different subsets of rows within the table use `partition by`
```sql
select *, sum(salary) -- sum is aggregate function applied to windows using over
over(partition by job order by salary desc) as total_job_salary
from emp;
```

* Rank employees based on salary
```sql
select *, 
row_number() over(partition by job order by salary) as partition_row_number, -- won't allow ties
rank() over(partition by job order by salary) as rank_row -- skips next rank if there is a tie, dense_rank doesn't do this
from emp;
```

* Create column with the name of the person with third highest salary in each partition
```sql
select *, 
nth_value(name, 3) over(partition by job order by salary range between unbounded preceding and unbounded following) as third_highest,
-- nth_value has to work on frames created by 'range between'
from emp;
```

* create quartile, or any quintile column
```sql
select *,
ntile(4) over(order by salary) as quartile -- no partition by in this over(), can use partition by if you want quintiles for different groups
from emp; -- every row in the table will be assigned a quartile based on the entire table
```

* create column that has salary of previous row, and the difference between the salaries
```sql
select *,
lag(salary, 1) over(partition by job order by salary) as sal_prev,
salary - lag(salary, 1) over(partition by job order by salary) as sal_diff
from emp;
```

```sql
-- employee table and department table, get employee with highest salary in each department
-- output employee name, department name, and salary
```
## Many to many
* requires third table to create relationship between the tables with a many to many relationship