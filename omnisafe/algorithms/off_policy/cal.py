# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Conservative Augmented Lagrangian algorithm."""


import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintEnsembledActorQCritic



@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class CAL(SAC):
    """The Conservative Augmented Lagrangian (CAL) algorithm.

    References:
        - Title: Off-Policy Primal-Dual Safe Reinforcement Learning
        - Authors: Zifan Wu, Bo Tang, Qian Lin, Chao Yu, Shangqin Wang, Qianlong Xie, Xingxing Wang, Dong Wang.
        - URL: `CAL <https://arxiv.org/abs/2401.14758>`_
    """

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_model(self) -> None:
        """Initialize the model.

        The ``num_critics`` in ``critic`` configuration must be 2.
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 2
        # The efficient implementation of ensemble only applies to cost critic here, 
        # but can also apply to reward critic to accelerate training, just by 
        # modifying the class `ActorQCritic` in `actor_q_critic.py` and 
        # importing `QEnsemble` from `q_critic.py`
        self._actor_critic = ConstraintEnsembledActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init_log(self) -> None:
        """Log the CAL specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Value/violation')
        self._logger.register_key('Value/conservative_qc')

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.
        """
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_iters`` times.
        """
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)
            
            violation = self._logger.get_stats('Value/violation')[0]
            conservative_qc = self._logger.get_stats('Value/conservative_qc')[0]
            if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
                if self._cfgs.algo_cfgs.ALM_dual_update:
                    self._lagrange.update_augmented_lagrange_multiplier(violation, self._cfgs.algo_cfgs.c)
                else:
                    self._lagrange.update_lagrange_multiplier(conservative_qc)
            self._logger.store(
                {
                    'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
                },
            )
    
    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update cost critic.

        - Get the TD loss of cost critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_c = self._actor_critic.target_cost_critic(next_obs, next_action) # shape(E, B, 1)
            if self._cfgs.algo_cfgs.intrgt_max:
                qc_idxs = np.random.choice(self._cfgs.model_cfgs.num_cost_critics, 2)
                next_qc_random_max = next_q_value_c[qc_idxs].max(dim=0, keepdim=True).values
                next_q_value_c = next_qc_random_max.repeat(self._cfgs.model_cfgs.num_cost_critics, 1, 1)
            target_q_value_c = cost.repeat(self._cfgs.model_cfgs.num_cost_critics, 1, 1) + \
                    self._cfgs.algo_cfgs.gamma * (1 - done).repeat(self._cfgs.model_cfgs.num_cost_critics, 1, 1) * next_q_value_c
        q_value_c = self._actor_critic.cost_critic(obs, action)
        loss = nn.functional.mse_loss(q_value_c, target_q_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q_value_c.mean().item(),
            },
        )

    def _update_actor(
        self,
        obs: torch.Tensor,
    ) -> None:
        """Update actor and alpha if ``auto_alpha`` is True.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
        """
        super()._update_actor(obs)

        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                action = self._actor_critic.actor.predict(obs, deterministic=False)
                log_prob = self._actor_critic.actor.log_prob(action)
            alpha_loss = -self._log_alpha * (log_prob + self._target_entropy).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._logger.store(
                {
                    'Loss/alpha_loss': alpha_loss.mean().item(),
                },
            )
        self._logger.store(
            {
                'Value/alpha': self._alpha,
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SACLag is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \lambda Q^C (s, \pi (s))

        where :math:`Q^V` is the min value of two reward critic networks outputs, :math:`Q^C` is the
        value of cost critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        loss_q_r_1, loss_q_r_2 = self._actor_critic.reward_critic(obs, action)
        loss_r = self._alpha * log_prob - torch.min(loss_q_r_1, loss_q_r_2)
        loss_q_cs = self._actor_critic.cost_critic(obs, action) # shape(E, B, 1)
        # conservative cost optimization
        qc_std, qc_mean = torch.std_mean(loss_q_cs, dim=0)
        loss_q_c = qc_mean + self._cfgs.algo_cfgs.k * qc_std
        # local policy convexification (gradient rectification)
        violation = torch.mean(self._lagrange.cost_limit - loss_q_c.detach())
        rect = self._cfgs.algo_cfgs.c * violation
        rect = torch.clamp(self.rect.detach(), max=self._lagrange.lagrangian_multiplier.item())
        loss_c = (self._lagrange.lagrangian_multiplier.item() - rect) * loss_q_c

        self._logger.store(
            {
                'Value/violation': violation.item(),
                'Value/conservative_qc': loss_q_c.detach().mean().item(),
            },
        )

        # aa

        return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier.item())

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )
