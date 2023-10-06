# Traffic and Vehicle Analysis on Controller Area Network (CAN) and GPS Data

The NGSIM project was a seminal traffic data collection experiment conducted at several locations. Read about the project at: https://datahub.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/#trajectory-data.
The US-101 trajectory data is a dataset collected on the US 101 freeway in California. It is available at the following link: https://data.transportation.gov/api/views/8ect-6jqj/files/bf7ca201-c7b3-4dc0-8f77-578f7794fcee?download=true&filename=US-101-LosAngeles-CA.zip The download includes multiple datasets.

There are several important fields:  
• **Vehicle_ID:** Vehicle identification number (ascending by time of entry into section). Global_Time: contains the time of the trajectory as a UNIX timestamp. For example, the first timestamp is 1118847869000, which converts to Wed Jun 15 2005 15:04:29 GMT+0000.  

• **Local_X:** Lateral (X) coordinate of the front center of the vehicle in feet with respect to the left-most edge of the section in the direction of travel.  

• **Local_Y:** Longitudinal (Y) coordinate of the front center of the vehicle in feet with respect to the entry edge of the section in the direction of travel.  

• **v_Length:** length of the vehicle in ft  

• **v_Width:** width of the vehicle in ft  

• **v_Vel:** an estimate of the vehicle velocity in ft/sec 

• **v_Acc:** an estimate of the vehicle acceleration in ft/sec2 

• **Lane_ID:** Current lane position of vehicle. Lane 1 is farthest left lane; lane 5 is farthest right lane. Lane 6 is the auxiliary lane between Ventura Boulevard on-ramp and the Cahuenga Boulevard off-ramp. Lane 7 is the on-ramp at Ventura Boulevard, and Lane 8 is the off-ramp at Cahuenga Boulevard.  

• **Preceding:** Vehicle Id of the lead vehicle in the same lane. A value of ’0’ represents no preceding vehicle - occurs at the end of the study section and off-ramp due to the fact that only complete trajectories were recorded by this data collection effort (vehicles already in the section at the start of the study period were not recorded).  

• **Following:** Vehicle Id of the vehicle following the subject vehicle in the same lane. A value of ’0’ represents no following vehicle - occurs at the beginning of the study section and onramp due to the fact that only complete trajectories were recorded by this data collection effort (vehicle that did not traverse the downstream boundaries of the section by the end of the study period were not recorded).
