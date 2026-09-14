terraform {
    required_providers {
        azurerm = {
            source = "hashicorp/azurerm"
            version = "~> 4.0"
        }
    }
}

provider "azurerm" {
    features {}
}

resource "azurerm_resource_group" "rg" {
    name     = "igann-mlops-rg"
    location = "germanywestcentral"
}

resource "azurerm_container_registry" "acr" {
    name                = "igannmlopsacr"
    resource_group_name = azurerm_resource_group.rg.name
    location            = azurerm_resource_group.rg.location
    sku                 = "Basic"
    admin_enabled       = true
}

resource "azurerm_kubernetes_cluster" "aks" {
    name                = "igann-mlops-aks"
    resource_group_name = azurerm_resource_group.rg.name
    location            = azurerm_resource_group.rg.location
    dns_prefix          = "igannmlops"

    default_node_pool {
        name        = "default"
        node_count  = 1
        vm_size     = "standard_D2s_v7"
    }

    identity {
        type = "SystemAssigned"
    }
}

resource "azurerm_role_assignment" "aks_acr_pull" {
    scope                = azurerm_container_registry.acr.id
    role_definition_name = "AcrPull"
    principal_id         = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
}
