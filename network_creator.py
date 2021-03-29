# Copyright 2021 Loro Francesco
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import larq as lq
import alexnet
import binary_densenet
import binary_densenet37_dilated
import binary_resnet_e18
import birealnet
import quicknet

if __name__ == "main":
  network = alexnet.AlexNet()
  _, model = network.build()
  model._name = "AlexNet"
  lq.models.summary(model)

  network = birealnet.BirealNet()
  _, model = network.build()
  model._name = "BiRealNet"
  lq.models.summary(model)

  network = binary_densenet.DenseNet(28)
  _, model = network.build()
  model._name = "DenseNet_E28"
  lq.models.summary(model)

  network = binary_densenet.DenseNet(37)
  _, model = network.build()
  model._name = "DenseNet_E37"
  lq.models.summary(model)

  network = binary_densenet.DenseNet(45)
  _, model = network.build()
  model._name = "DenseNet_E45"
  lq.models.summary(model)

  network = quicknet.QuickNet("")
  _, model = network.build()
  model._name = "QuickNet"
  lq.models.summary(model)

  network = quicknet.QuickNet("small")
  _, model = network.build()
  model._name = "QuickNet Small"
  lq.models.summary(model)

  network = quicknet.QuickNet("large")
  _, model = network.build()
  model._name = "QuickNet Large"
  lq.models.summary(model)

  network = binary_densenet37_dilated.DenseNetE37Dilated()
  _, model = network.build()
  model._name = "DenseNet_E37 Dilated"
  lq.models.summary(model)
