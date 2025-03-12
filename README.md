Installation
============
Requires Python, Dash, Dash bootstrap components, networkx and jsonpickle. Uses plotly graph objects, but these come with Dash.

Running
=======
Run with `python ui.py <configuration file>`

A sample configuration file is included.


Configuration
=============

The configuration file contains the following sections, described by a keyword that *must* appear in the file.

```
TANKS
CONNECTIONS
ALARMS
RepairTeams
SCORING
BREAKS
MillisPerTick
```
BREAKS are optional.

Tanks
=====

This section describes tanks. The first two entries should be

```
Source: <ID> <POSITION>
```
and
```
Sink: <ID> <POSITION>
```

ID is a string and the optional POSITION parameter (made up of two non-negative numbers) describes the position of these tanks in the GUI.

Any number of tanks can then follow with the following form
```
Tank: <ID> <CAPACITY> <CAN OVERFLOW> <VISIBLE> <POSITION>
```
<ID> is again a string, <CAPACITY> describes the amount of liquid the tank can hold. <CAN OVERFLOW> is a boolean (True or False). If true, and the tank's capacity is exceeded, the simulation will end. If false, once at capacity, no more liquid can enter the tank. <VISIBLE> states whether the amount the tank currently holds will appear on the GUI.

Connections
==========

There are 4 types of connections. The general format of a connection is
```
<CONNECTION TYPE> <SOURCE> <SINK> <FLOW DETAILS> <BREAK DETAILS>
```
<CONNECTION TYPE> is one of `Pipe:`, `Valve:` `Tap:` or `RandomPipe`.
<SOURCE> and <SINK> are Tank IDs.

Pipes
-----

The flow details for a pipe are simply the amount of liquid it can carry, i.e., a single number.

Valves
------
Valves are boolean and so flow details are the minimum and maximum flow they can carry.

Taps
----
Taps allow for varied flow. Again, flow details are the minimum and maximum flow they can carry.

Random pipes
------------
Random pipes also have minimum and maximum flows. At each time step in the simulation, a random value between these flows will be selected.

Break information
-----------------
This consists of the likelihood of the connection breaking, the minimum and maximum times taken to repair the break, and the minimum and maximum flow rate that will occur when broken. A number will be selected uniformly between these two rates.

Alarms
=======
Alarms consist of the following
```
Alarm <Tank ID> <Level> <Type>
```
Type is either `Low` or `High`. A low alarm will trigger whenever the tank is below level, while a high alarm will trigger when the tank is above the level.

Repairs
=======
The `RepairTeam` is followed by a single number specifying how many repair teams exist.

Scoring
=======
This section consists of lines of the form
```
ScoreCondition <ID> <Level> <Type>
```

These function similarly to alarms, but when all score conditions are met, the score will increase by 1 at that timestep.

Breaks
======
One can also specify breaks manually. These take the form
```
Break <Connection ID> <Time> <Repair time> <Flow>
```
Time specifies when in the simulation the break will occur, while flow specifies how much liquid will flow through the connector when the break occurs.

MillisPerTick
=============
Followed by a number, this states how many milliseconds occur between each tick of simulation time.

Flow model
==========
A connection has an in- and out- part which can hold up to the connection's max flow in liquid. Flow occurs in 3 stages. First, a tank puts as much liquid as it can into the in part of a connection. If insufficient liquid exists, all the liquid available will go into a subset of connections (e.g., if 5 units are available and there are 4 connections with 2 each, 2 connections will recieve 2 units and 1 will recieve 1 unit, with the last receiving 0).

The flow rate then determines how much liquid is moved from the in- side to the out- side. Finally, the out-side is emptied into the receiving tank, again, emptying connections first if possible.

Sinks can absorb infinite liquid, sources can produce infinite liquid.

