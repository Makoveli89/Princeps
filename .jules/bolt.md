## 2024-05-23 - Eliminating N+1 Queries in Workspace Listing
**Learning:** Scalar subqueries in SQLAlchemy (`correlate().scalar_subquery()`) are an extremely effective way to aggregate counts from related tables in a single query, avoiding the Cartesian product issues of joins and the N+1 problem of loops.
**Action:** Always look for loops over parent entities that query child counts and replace them with correlated subqueries.
