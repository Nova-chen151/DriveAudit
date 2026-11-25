"""Onsite-inspired greedy token selection with dynamics and observed-road checks.

Selects intact CAT-K vocabulary tokens. No projection, deletion, GT target, or
new trajectory synthesis. Infeasible choices remain explicit in diagnostics.
"""
import torch


def vehicle_token_mask(tokens):
    center = tokens.mean(-2)
    velocity = torch.diff(center, dim=-2) / .1
    acceleration = torch.linalg.vector_norm(torch.diff(velocity, dim=-2) / .1, dim=-1)
    edge = tokens[..., 0, :] - tokens[..., 3, :]
    heading = torch.atan2(edge[..., 1], edge[..., 0])
    turn = (torch.diff(heading, dim=-1) + torch.pi) % (2*torch.pi) - torch.pi
    # Reference Onsite export uses 6 m/s², 60 m/s and 0.20 rad per 0.1 s.
    return ((acceleration <= 6).all(-1) & (torch.linalg.vector_norm(velocity, dim=-1) <= 60).all(-1)
            & (turn.abs() <= .20).all(-1))


class ConstrainedGreedy:
    def __init__(self, context, agent, pool_size=64, collision_checks=True):
        device = agent['position'].device
        self.vehicle = agent['type'] == 0
        self.shape = agent['shape'][:, :2]
        self.previous_velocity = agent['velocity'][:, 10].clone()
        self.pool_size = pool_size
        self.collision_checks = collision_checks
        self.map_labels = torch.as_tensor(context.drivable_area, device=device, dtype=torch.int8)
        self.map_labels = self.map_labels.masked_fill(~torch.as_tensor(context.observed_area, device=device), -1)
        self.origin = torch.as_tensor(context.extent[[0, 2]], device=device, dtype=torch.float32)
        self.resolution = context.resolution
        # ponytail: fixed footprint grid bounds GPU cost; native-time QC separately uses 0.25 m interior samples.
        self.local_grid = torch.cartesian_prod(torch.linspace(-.5, .5, 11, device=device),
                                               torch.linspace(-.5, .5, 7, device=device))
        start = agent['road_start_position'][:, None, None]
        heading = agent['road_start_heading'][:, None, None]
        idx = torch.arange(len(self.vehicle), device=device)
        fraction, coverage = self._footprints(start, heading, idx)
        self.allow_outside = fraction[:, 0, 0]
        self.protected = self.vehicle & (coverage[:, 0, 0] >= .8)
        self.stats = dict(token_decisions=0, reranked_decisions=0, expanded_search_decisions=0,
                          road_constraint_unsatisfied=0, seam_acceleration_relaxed=0,
                          braking_lookahead_unsatisfied=0,
                          collision_reranked_decisions=0, collision_constraint_unsatisfied=0,
                          map_unprotected_agents=int((self.vehicle & ~self.protected).sum()),
                          deleted_agents=0, projected_states=0)
        self.centers = None

    def _footprints(self, position, heading, indices):
        local = self.local_grid[None, None, None] * self.shape[indices, None, None, None]
        cosine, sine = heading.cos()[..., None], heading.sin()[..., None]
        x = local[..., 0]*cosine-local[..., 1]*sine+position[..., 0, None]
        y = local[..., 0]*sine+local[..., 1]*cosine+position[..., 1, None]
        ij = torch.floor((torch.stack((x, y), dim=-1)-self.origin)/self.resolution).long()
        inside = ((ij >= 0) & (ij < torch.tensor(self.map_labels.shape, device=ij.device))).all(-1)
        labels = self.map_labels[ij[..., 0].clamp(0, self.map_labels.shape[0]-1),
                                 ij[..., 1].clamp(0, self.map_labels.shape[1]-1)].masked_fill(~inside, -1)
        observed = (labels != -1).sum(-1)
        return (labels == 0).sum(-1)/observed.clamp_min(1), observed/labels.shape[-1]

    def _poses(self, candidates, indices, pos_now, head_now):
        local = self.centers[indices[:, None], candidates, 1:]
        cosine, sine = head_now[indices, None, None].cos(), head_now[indices, None, None].sin()
        position = torch.stack((local[..., 0]*cosine-local[..., 1]*sine,
                                local[..., 0]*sine+local[..., 1]*cosine), dim=-1)+pos_now[indices, None, None]
        heading = self.headings[indices[:, None], candidates, 1:]+head_now[indices, None, None]
        return position, heading

    def _evaluate(self, candidates, indices, pos_now, head_now):
        position, heading = self._poses(candidates, indices, pos_now, head_now)
        outside, coverage = self._footprints(position, heading, indices)
        road = ((outside-self.allow_outside[indices, None, None]).clamp_min(0)
                + (.8-coverage).clamp_min(0)).amax(-1)
        road = road.masked_fill(~self.protected[indices, None], 0)
        # Check space to brake at the reference 6 m/s², along the token's terminal curvature.
        # This is a candidate test only; these lookahead points are never exported as predictions.
        delta = position[:, :, -1]-position[:, :, -2]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        tangent = torch.atan2(delta[..., 1], delta[..., 0])
        turn = (heading[:, :, -1]-heading[:, :, -2]+torch.pi) % (2*torch.pi)-torch.pi
        curvature = turn/distance.clamp_min(.01)
        travel = (distance/.1).square()[..., None]/12*torch.linspace(.125, 1, 8, device=position.device)
        angle = curvature[..., None]*travel
        chord = travel*torch.sinc(angle/(2*torch.pi))
        direction = tangent[..., None]+angle/2
        stop_position = position[:, :, -1, None]+torch.stack((chord*direction.cos(), chord*direction.sin()), dim=-1)
        stop_heading = heading[:, :, -1, None]+angle
        outside, coverage = self._footprints(stop_position, stop_heading, indices)
        braking = ((outside-self.allow_outside[indices, None, None]).clamp_min(0)
                   + (.8-coverage).clamp_min(0)).amax(-1).masked_fill(~self.protected[indices, None], 0)
        velocity = (position[:, :, 0]-pos_now[indices, None])/.1
        seam = (torch.linalg.vector_norm(velocity-self.previous_velocity[indices, None], dim=-1)-.6).clamp_min(0)
        seam = seam.masked_fill(~self.vehicle[indices, None], 0)
        good = self.allowed[indices[:, None], candidates]
        return road.masked_fill(~good, torch.inf), seam.masked_fill(~good, torch.inf), braking.masked_fill(~good, torch.inf)

    def _collision_cost(self, position, heading, indices, others, other_heading):
        # Full observed OBB dimensions, all five 10 Hz states; no future annotations.
        u = torch.stack((heading.cos(), heading.sin()), -1)[:, :, None]
        v = torch.stack((-heading.sin(), heading.cos()), -1)[:, :, None]
        ou = torch.stack((other_heading.cos(), other_heading.sin()), -1)[None, None]
        ov = torch.stack((-other_heading.sin(), other_heading.cos()), -1)[None, None]
        u, v, ou, ov = torch.broadcast_tensors(u, v, ou, ov)
        axes = torch.stack((u, v, ou, ov), -2)
        delta = others[None, None]-position[:, :, None]
        distance = (delta[..., None, :]*axes).sum(-1).abs()
        radius = ((u[..., None, :]*axes).sum(-1).abs()*self.shape[indices, 0, None, None, None, None]/2
                  +(v[..., None, :]*axes).sum(-1).abs()*self.shape[indices, 1, None, None, None, None]/2
                  +(ou[..., None, :]*axes).sum(-1).abs()*self.shape[None, None, :, 0, None, None]/2
                  +(ov[..., None, :]*axes).sum(-1).abs()*self.shape[None, None, :, 1, None, None]/2)
        depth = (radius-distance).amin(-1).clamp_min(0)
        self_pair = indices[:, None] == torch.arange(len(others), device=indices.device)[None]
        depth = depth.masked_fill(self_pair[:, None, :, None], 0)
        return (depth > 1e-6).sum((-1,-2)) + depth.clamp_max(1).sum((-1,-2))/(depth.shape[-1]*depth.shape[-2]+1)

    def _avoid_collisions(self, chosen, candidates, scores, indices, pos_now, head_now):
        position, heading = self._poses(chosen[:, None], indices, pos_now, head_now)
        position, heading = position[:, 0], heading[:, 0]
        # ponytail: two sequential sweeps over top-64 tokens; residual conflicts need wider joint search.
        for _ in range(2):
            cost = self._collision_cost(position[:, None], heading[:, None], indices, position, heading)[:, 0]
            for actor in torch.where(cost > 0)[0].tolist():
                ix = indices[actor:actor+1]
                pool = torch.cat((chosen[actor:actor+1], candidates[actor]))[None]
                p, h = self._poses(pool, ix, pos_now, head_now)
                c = self._collision_cost(p, h, ix, position, heading)[0]
                r, s, b = self._evaluate(pool, ix, pos_now, head_now)
                usable = self.allowed[actor, pool[0]] & (r[0] <= r[0, 0]+1e-7)
                usable &= (s[0] <= s[0, 0]+1e-5) & (b[0] <= b[0, 0]+1e-7)
                c = c.masked_fill(~usable, torch.inf)
                usable &= c <= c.min()+1e-7
                best = scores[actor, pool[0]].masked_fill(~usable, -torch.inf).argmax()
                self.stats['collision_reranked_decisions'] += int(chosen[actor] != pool[0, best])
                chosen[actor] = pool[0, best]
                position[actor], heading[actor] = p[0, best], h[0, best]
        remaining = self._collision_cost(position[:, None], heading[:, None], indices, position, heading)
        self.stats['collision_constraint_unsatisfied'] += int((remaining > 0).sum())
        return chosen

    def __call__(self, token_traj_all, next_token_logits, pos_now, head_now, **unused):
        # Deliberately never reads tokenized future GT fields supplied by the official decoder.
        n, count = next_token_logits.shape
        indices = torch.arange(n, device=next_token_logits.device)
        if self.centers is None:
            self.centers = token_traj_all.mean(-2)
            edge = token_traj_all[..., 0, :]-token_traj_all[..., 3, :]
            self.headings = torch.atan2(edge[..., 1], edge[..., 0])
            self.allowed = torch.ones((n, count), device=next_token_logits.device, dtype=torch.bool)
            if self.vehicle.any():
                representative = torch.where(self.vehicle)[0][0]
                allowed = vehicle_token_mask(token_traj_all[representative])
                if not allowed.any(): raise ValueError('No dynamically valid vehicle vocabulary tokens')
                # torch 2.4 deterministic CUDA index_put cannot broadcast this boolean-index assignment.
                self.allowed = torch.where(self.vehicle[:, None], allowed[None], self.allowed)
                self.stats['vehicle_vocabulary_tokens_excluded'] = int((~allowed).sum())
        scores = next_token_logits.detach().masked_fill(~self.allowed, -torch.inf)
        candidates = scores.topk(min(self.pool_size, count), dim=-1, sorted=True).indices
        road, seam, braking = self._evaluate(candidates, indices, pos_now, head_now)
        feasible = (road <= 1e-7) & (seam <= 1e-5) & (braking <= 1e-7)
        ranked = scores.gather(1, candidates).masked_fill(~feasible, -torch.inf)
        chosen = candidates.gather(1, ranked.argmax(-1, keepdim=True)).squeeze(1)
        failed = torch.where(~feasible.any(-1))[0]
        self.stats['expanded_search_decisions'] += len(failed)
        # Expand only failed actors; one actor at a time keeps peak map-query memory bounded.
        for actor in failed.tolist():
            ix = indices[actor:actor+1]
            all_candidates = torch.arange(count, device=indices.device)[None]
            r, s, b = self._evaluate(all_candidates, ix, pos_now, head_now)
            r, s, b = r[0], s[0], b[0]
            usable = self.allowed[actor] & (r <= r.min()+1e-7)
            min_seam = s.masked_fill(~usable, torch.inf).min()
            usable &= s <= min_seam+1e-5
            min_braking = b.masked_fill(~usable, torch.inf).min()
            usable &= b <= min_braking+1e-7
            chosen[actor] = scores[actor].masked_fill(~usable, -torch.inf).argmax()
            self.stats['road_constraint_unsatisfied'] += int(r[chosen[actor]] > 1e-7)
            self.stats['seam_acceleration_relaxed'] += int(s[chosen[actor]] > 1e-5)
            self.stats['braking_lookahead_unsatisfied'] += int(b[chosen[actor]] > 1e-7)
        if self.collision_checks and n > 1:
            chosen = self._avoid_collisions(chosen, candidates, scores, indices, pos_now, head_now)
        self.stats['token_decisions'] += n
        self.stats['reranked_decisions'] += int((chosen != next_token_logits.argmax(-1)).sum())
        centers = self.centers[indices, chosen]
        velocity = (centers[:, -1]-centers[:, -2])/.1
        self.previous_velocity = torch.stack((velocity[:, 0]*head_now.cos()-velocity[:, 1]*head_now.sin(),
                                              velocity[:, 0]*head_now.sin()+velocity[:, 1]*head_now.cos()), dim=-1)
        return chosen, token_traj_all[indices, chosen]
