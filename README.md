# Drone Interceptors 

2D simulation of a drone swarm intercepting missiles — with **no communication between drones**.

## The Problem

A swarm of interceptor drones defends a base from a wave of incoming missiles.
Two hard constraints make this interesting:

- **Each drone is destroyed on impact.** Wasting two drones on the same
  missile means one less defender for the next wave.
- **The drones can't communicate.** No radio, no central command — no one
  can say "you take the left one, I'll take this one."

So the core question is: **which drone goes after which missile?**

If every drone simply chases its nearest missile, several drones lock onto
the same target while others go unchallenged. That's the failure mode this
project sets out to solve.

## The Logic

Every drone sees the same map — all missile positions, all allied drone
positions — and runs the exact same rule:

1. For each missile, compute which drone is closest to it
   (ties broken by lowest drone ID).
2. If the closest drone is *me*, that missile is mine.
3. Among "my" missiles, take the nearest one.

Same data in, same rule applied → every drone independently reaches the
**same global assignment**. Full coordination with zero messages exchanged.

### Flying to the right place

Chasing a missile's current position always leaves you behind it. So each
drone instead solves the interception equation: *at what future point will
my path and the missile's path cross, and can I get there in time?* It then
flies straight to that meeting point.

### While waiting

Drones with no assigned target don't hover in place — they fall back to a
defensive formation between the remaining missiles and the base, each at
its own station, already on the way for whatever comes next.

## Features

- Decentralized assignment: "closest drone wins", ID tie-breaking
- Predictive interception (quadratic intercept solver)
- One-for-one sacrifice: drones explode on interception
- Defensive formation for idle drones
- Randomized missile spawns on every run

## Requirements

```bash
pip install numpy matplotlib
